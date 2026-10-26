"""Train a quantile forecaster on a chronological train/validation split and save a self-contained checkpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import prepare_training_data
from data.preprocess import FEATURE_COLUMNS, prepare_frame
from models import MODEL_TYPES, build_model, count_parameters
from models.layers import PinballLoss
from models.predictor import PricePredictor
from utils.helpers import DEFAULT_CONFIG_PATH, cli_path, get_device, load_config, resolve_path, set_seed
from utils.logger import get_logger

logger = get_logger(__name__)


def run_epoch(model, loader, criterion, device, optimizer=None, max_grad_norm=1.0) -> float:
    """One pass over ``loader``; trains when an optimizer is given, otherwise only evaluates."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, total_samples = 0.0, 0
    with torch.set_grad_enabled(is_training):
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device).unsqueeze(-1)
            loss = criterion(model(sequences), labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
    return total_loss / total_samples


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=8, scheduler=None):
    """Early-stop on validation loss (never on training loss) and restore the best weights."""
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("Epoch %3d/%d | train loss %.4f | val loss %.4f", epoch, num_epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping: no validation improvement for %d epochs", patience)
                break

    model.load_state_dict(best_state)
    return history


def validate_quantiles(quantiles) -> list:
    quantiles = [float(q) for q in quantiles]
    if quantiles != sorted(set(quantiles)) or not all(0 < q < 1 for q in quantiles) or 0.5 not in quantiles:
        raise ValueError("quantiles must be strictly increasing values in (0, 1) that include 0.5")
    rounded = [round(q, 6) for q in quantiles]
    if not any(round(1 - q, 6) in rounded for q in rounded if q < 0.5):
        raise ValueError("quantiles need at least one symmetric pair, such as 0.05 and 0.95, to form an interval")
    return quantiles


def train(config: dict, device: str = "auto"):
    set_seed(config["seed"])
    device = get_device(device)

    frame = prepare_frame(resolve_path(config["data_path"]), horizon=config["horizon"])
    data = prepare_training_data(
        frame, config["seq_length"], config["val_ratio"], config["test_ratio"], horizon=config["horizon"]
    )
    for split in ("train", "val", "test"):
        indices = data.splits[split]
        logger.info(
            "%-5s %6d samples | %s -> %s", split, len(indices), frame.index[indices[0]], frame.index[indices[-1]]
        )

    train_loader = DataLoader(data.dataset("train"), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(data.dataset("val"), batch_size=config["batch_size"])

    quantiles = validate_quantiles(config["quantiles"])
    model = build_model(
        config["model_type"],
        input_size=len(FEATURE_COLUMNS),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_size=len(quantiles),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=max(1, config["patience"] // 3)
    )
    logger.info(
        "Training %s (%s parameters) on %s", config["model_type"].upper(), f"{count_parameters(model):,}", device
    )

    # The pinball loss is linear in the error, so the rare crash candles do not dominate the gradients.
    criterion = PinballLoss(quantiles).to(device)
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config["num_epochs"],
        device,
        config["patience"],
        scheduler,
    )

    predictor = PricePredictor(model, config, FEATURE_COLUMNS, data.feature_scaler, data.target_scaler, data.boundaries)
    predictor.save(resolve_path(config["model_path"]), history=history)
    logger.info("Saved checkpoint to %s", config["model_path"])
    return predictor, history


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="JSON config file")
    parser.add_argument("--data-path", help="OHLCV CSV to train on (overrides the config)")
    parser.add_argument("--model-path", help="Where to save the checkpoint (overrides the config)")
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES), help="Network architecture")
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(
        args.config,
        data_path=cli_path(args.data_path),
        model_path=cli_path(args.model_path),
        model_type=args.model_type,
        num_epochs=args.epochs,
    )
    train(config, device=args.device)


if __name__ == "__main__":
    main()
