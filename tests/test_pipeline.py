import json

import numpy as np
import pandas as pd

from scripts import evaluate as evaluate_script
from scripts import predict as predict_script
from scripts import train as train_script


def test_train_evaluate_predict_end_to_end(tiny_config, tmp_path):
    predictor, history = train_script.train(tiny_config, device="cpu")
    assert 1 <= len(history) <= tiny_config["num_epochs"]
    assert all(np.isfinite(epoch["val_loss"]) for epoch in history)

    plot_path = tmp_path / "plot.png"
    results = evaluate_script.evaluate(tiny_config["model_path"], device="cpu", plot_path=plot_path)
    assert results["samples"] > 0
    assert all(np.isfinite(results[name][metric]) for name in ("model", "naive") for metric in ("mae", "rmse", "mape"))
    assert plot_path.stat().st_size > 0
    assert "Naive (last close)" in evaluate_script.format_report(results)

    predictor, frame = predict_script.load_predictor_and_frame(tiny_config["model_path"], device="cpu")
    result = predict_script.forecast(predictor, frame)
    assert result["last_candle"] == frame.index[-1]
    assert result["target_candle"] == frame.index[-1] + pd.Timedelta(hours=1)
    assert result["predicted_close"] > 0

    history_frame = predict_script.prediction_history(predictor, frame)
    assert set(history_frame["split"]) == {"train", "val", "test", "forecast"}
    assert history_frame["actual_close"].isna().sum() == 1
    test_rows = history_frame[history_frame["split"] == "test"]
    assert len(test_rows) == results["samples"]


def test_command_line_entry_points(tiny_config, tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config))
    model_path = str(tmp_path / "cli_model.pt")

    train_script.main(
        ["--config", str(config_path), "--model-path", model_path, "--epochs", "1", "--model-type", "lstm"]
    )

    metrics_path = tmp_path / "metrics.json"
    evaluate_script.main(
        ["--model-path", model_path, "--split", "val", "--metrics-out", str(metrics_path), "--device", "cpu"]
    )
    metrics = json.loads(metrics_path.read_text())
    assert metrics["split"] == "val"
    assert metrics["model_type"] == "lstm"

    history_path = tmp_path / "history.csv"
    predict_script.main(["--model-path", model_path, "--history-out", str(history_path), "--device", "cpu"])
    output = capsys.readouterr().out
    assert "Directional accuracy" in output
    assert "Forecast for the" in output
    assert history_path.exists()
