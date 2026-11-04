from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train" / "train.py"


def load_train_module():
    """Import train/train.py without its heavy deps (they are only imported inside functions)."""
    spec = importlib.util.spec_from_file_location("train_entry", TRAIN_PY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wait_for(pred, timeout: float = 20.0, interval: float = 0.02):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    raise AssertionError("condition not met in time")


@pytest.fixture
def train_module():
    return load_train_module()
