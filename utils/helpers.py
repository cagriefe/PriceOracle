import json
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scripts" / "config.json"


def load_config(path=DEFAULT_CONFIG_PATH, **overrides) -> dict:
    """Load the JSON config and apply every override that is not ``None`` (e.g. CLI flags)."""
    with open(path) as f:
        config = json.load(f)
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def resolve_path(path) -> Path:
    """Resolve relative paths against the project root, so config paths work from any directory."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def cli_path(path):
    """Make a user-supplied CLI path absolute relative to the current directory (``None`` passes through)."""
    return None if path is None else str(Path(path).expanduser().resolve())


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (all devices) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Pick CUDA when available, otherwise CPU; any explicit device string (``cpu``, ``cuda``, ``mps``) wins."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
