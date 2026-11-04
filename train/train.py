"""GPU Run Agent training entrypoint: LoRA supervised fine-tuning with TRL + PEFT.

Runs inside the training image (train/Dockerfile) on any provider. It is deliberately standalone:
only the training libraries and the standard library, no `gpu_agent` import.

Input: a job spec as JSON, from `--spec path.json` or the GPU_AGENT_SPEC environment variable.
Unknown keys are rejected; missing keys take the defaults in DEFAULTS (kept in sync with
`gpu_agent.models.JobSpec` by a test).

Output: human-readable logs, plus one JSON object per line prefixed with EVENT_PREFIX:
    {"event": "start",   ...}   model, dataset, device, versions
    {"event": "metrics", ...}   step, loss, num_tokens, tokens_per_sec, peak_vram_gib, ...
    {"event": "done",    ...}   train_seconds, tokens, tokens_per_sec, peak VRAM, final_loss, hub_url
    {"event": "error",   ...}   error (traceback text)
Exit code is 0 on success, 1 on failure.

Hub: if `hub_repo` is set, HF_TOKEN must be in the environment. The repo is created up front (so a
bad token fails before any GPU time is spent), checkpoints are pushed with hub_strategy="checkpoint"
every `save_steps`, and the final adapter is pushed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any

EVENT_PREFIX = "GPU_AGENT_EVENT "

DEFAULTS: dict[str, Any] = {
    "base_model": "Qwen/Qwen2.5-0.5B",
    "dataset": "trl-lib/Capybara",
    "dataset_split": "train",
    "dataset_limit": None,
    "max_steps": None,
    "epochs": 1.0,
    "per_device_batch_size": 4,
    "grad_accum": 1,
    "learning_rate": 2e-4,
    "max_length": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "all-linear",
    "gradient_checkpointing": False,
    "logging_steps": 5,
    "save_steps": 50,
    "seed": 42,
    "hub_repo": None,
    "hub_private": True,
    "output_dir": "/tmp/gpu-agent-out",
}


def emit(event: str, **fields: Any) -> None:
    """Print one machine-readable event line. Values must be JSON-serialisable plain types."""
    print(EVENT_PREFIX + json.dumps({"event": event, "ts": round(time.time(), 3), **fields}, default=str),
          flush=True)


def load_spec(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", help="path to a JSON job spec (default: $GPU_AGENT_SPEC)")
    args = parser.parse_args(argv)
    if args.spec:
        with open(args.spec) as f:
            raw = json.load(f)
    elif os.environ.get("GPU_AGENT_SPEC"):
        raw = json.loads(os.environ["GPU_AGENT_SPEC"])
    else:
        raise SystemExit("no job spec: pass --spec path.json or set GPU_AGENT_SPEC")
    if not isinstance(raw, dict):
        raise ValueError("job spec must be a JSON object")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown job spec keys: {unknown}")
    return {**DEFAULTS, **raw}


def load_train_dataset(spec: dict[str, Any]):
    from datasets import load_dataset

    name, split = spec["dataset"], spec["dataset_split"]
    if name.endswith((".json", ".jsonl")):  # local file, mainly for tests
        ds = load_dataset("json", data_files=name, split=split)
    else:
        ds = load_dataset(name, split=split)
    if spec["dataset_limit"]:
        ds = ds.select(range(min(int(spec["dataset_limit"]), len(ds))))
    return ds


def make_metrics_callback(t_start: list[float]):
    import torch
    from transformers import TrainerCallback

    class MetricsCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            t_start[0] = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" not in logs:  # skip the final train summary; "done" covers it
                return
            elapsed = time.time() - t_start[0]
            toks = float(logs.get("num_tokens", 0) or 0)
            emit("metrics", step=state.global_step, max_steps=state.max_steps,
                 loss=logs.get("loss"), learning_rate=logs.get("learning_rate"),
                 grad_norm=logs.get("grad_norm"), num_tokens=int(toks),
                 tokens_per_sec=round(toks / elapsed, 1) if elapsed > 0 else None,
                 elapsed_s=round(elapsed, 1),
                 peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3)
                 if torch.cuda.is_available() else None)

    return MetricsCallback()


def train(spec: dict[str, Any]) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    import trl
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    hub_repo = spec["hub_repo"]
    emit("start", base_model=spec["base_model"], dataset=spec["dataset"], hub_repo=hub_repo,
         device=torch.cuda.get_device_name(0) if cuda else "cpu", bf16=bf16,
         versions={"torch": torch.__version__, "transformers": transformers.__version__,
                   "trl": trl.__version__, "peft": peft.__version__})

    hub_url = None
    if hub_repo:
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("hub_repo is set but HF_TOKEN is not in the environment")
        from huggingface_hub import HfApi

        # Fail fast on a bad token or namespace, before paying for model download and training.
        hub_url = str(HfApi().create_repo(hub_repo, private=bool(spec["hub_private"]), exist_ok=True))

    ds = load_train_dataset(spec)
    steps = spec["max_steps"]
    cfg = SFTConfig(
        output_dir=spec["output_dir"],
        max_steps=int(steps) if steps else -1,
        num_train_epochs=float(spec["epochs"]),
        per_device_train_batch_size=int(spec["per_device_batch_size"]),
        gradient_accumulation_steps=int(spec["grad_accum"]),
        learning_rate=float(spec["learning_rate"]),
        max_length=int(spec["max_length"]),
        gradient_checkpointing=bool(spec["gradient_checkpointing"]),
        bf16=bf16,
        logging_steps=int(spec["logging_steps"]),
        save_strategy="steps" if hub_repo else "no",
        save_steps=int(spec["save_steps"]),
        save_total_limit=1,
        seed=int(spec["seed"]),
        report_to="none",
        disable_tqdm=True,  # progress bars use \r and flood line-based log capture
        push_to_hub=bool(hub_repo),
        hub_model_id=hub_repo,
        hub_strategy="checkpoint",
        hub_private_repo=bool(spec["hub_private"]),
        use_cpu=not cuda,
    )
    t_start = [time.time()]
    trainer = SFTTrainer(
        model=spec["base_model"], args=cfg, train_dataset=ds,
        peft_config=LoraConfig(r=int(spec["lora_r"]), lora_alpha=int(spec["lora_alpha"]),
                               lora_dropout=float(spec["lora_dropout"]),
                               target_modules=spec["lora_target_modules"], task_type="CAUSAL_LM"),
        callbacks=[make_metrics_callback(t_start)],
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    train_seconds = time.time() - t_start[0]

    trainer.save_model(spec["output_dir"])  # adapter weights + config only (PEFT)
    if hub_repo:
        trainer.push_to_hub(commit_message="gpu-agent: final LoRA adapter")

    history = trainer.state.log_history
    toks = max((float(h.get("num_tokens", 0) or 0) for h in history), default=0.0)
    losses = [h["loss"] for h in history if "loss" in h]
    return {
        "train_seconds": round(train_seconds, 1),
        "steps": trainer.state.global_step,
        "tokens": int(toks),
        "tokens_per_sec": round(toks / train_seconds, 1) if train_seconds > 0 else None,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None,
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3) if cuda else None,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "adapter_dir": spec["output_dir"],
        "hub_repo": hub_repo,
        "hub_url": hub_url,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        spec = load_spec(argv)
        result = train(spec)
    except SystemExit:
        raise
    except BaseException:
        emit("error", error=traceback.format_exc())
        return 1
    emit("done", **result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
