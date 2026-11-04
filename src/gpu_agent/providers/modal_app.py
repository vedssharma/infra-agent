"""The Modal app: training image + one GPU function that runs a job and reports through a modal.Dict.

Deployed once (and redeployed on every launch, which is a no-op when nothing changed) by
`ModalProvider`; each launch then picks its GPU and TTL with `Function.with_options`.

This module is imported both locally and inside the container, so it only imports `modal`, the
standard library, and (inside the function) the stdlib-only `gpu_agent.runner`.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "gpu-agent"
FUNCTION_NAME = "run_job"
STORE_NAME = "gpu-agent-runs"      # modal.Dict holding run meta, state and log chunks
HF_SECRET_NAME = "gpu-agent-hf"    # Modal secret with HF_TOKEN (fine-grained, write to one namespace)
DEFAULT_GPU = "A10"
DEFAULT_TIMEOUT_S = 30 * 60        # every launch overrides this; it only bounds a stray direct call
TRAIN_PYTHON = "/opt/venv/bin/python"  # training venv from train/Dockerfile, isolated from Modal's deps
TRAIN_CMD = [TRAIN_PYTHON, "/opt/train/train.py"]


def _train_dir() -> Path:
    if os.environ.get("GPU_AGENT_TRAIN_DIR"):
        return Path(os.environ["GPU_AGENT_TRAIN_DIR"])
    here = Path(__file__).resolve()
    # src/gpu_agent/providers/modal_app.py -> repo root is three levels above the package dir.
    return here.parents[3] / "train" if len(here.parents) > 3 else Path("/opt/train")


TRAIN_DIR = _train_dir()

image = (
    modal.Image.from_dockerfile(TRAIN_DIR / "Dockerfile", context_dir=TRAIN_DIR)
    .add_local_python_source("gpu_agent")  # mounted at container start; code edits need no rebuild
)

app = modal.App(APP_NAME)


@app.function(image=image, gpu=DEFAULT_GPU, timeout=DEFAULT_TIMEOUT_S, retries=0,
              secrets=[modal.Secret.from_name(HF_SECRET_NAME)])
def run_job(spec: dict, run_id: str) -> dict:
    from gpu_agent.runner import run_job as run

    store = modal.Dict.from_name(STORE_NAME, create_if_missing=True)
    return run(spec, run_id, store, TRAIN_CMD)
