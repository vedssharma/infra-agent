"""M0 spike: 20-step TRL + PEFT LoRA smoke run on Modal; records peak VRAM and throughput.

    uv run --extra spikes modal run spikes/train_smoke.py
Public model + dataset, no keys needed. A10 (24 GB) for ~$1.10/hr; timeout 15 min enforced by Modal.
"""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "transformers", "trl", "peft", "datasets", "accelerate")
app = modal.App("gpu-agent-train-smoke", image=image)

MODEL = "Qwen/Qwen2.5-0.5B"
STEPS, BATCH, MAXLEN = 20, 4, 512


@app.function(gpu="A10", timeout=900)
def train() -> dict:
    try:
        return _train()
    except Exception:
        import traceback

        return {"error": str(traceback.format_exc())}


def _train() -> dict:
    import json
    import time

    import torch
    import transformers
    import trl
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset("trl-lib/Capybara", split=f"train[:{STEPS * BATCH * 2}]")
    cfg = SFTConfig(output_dir="/tmp/out", max_steps=STEPS, per_device_train_batch_size=BATCH, max_length=MAXLEN,
                    bf16=True, logging_steps=5, save_strategy="no", report_to="none")
    trainer = SFTTrainer(model=MODEL, args=cfg, train_dataset=ds,
                         peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    toks = max(float(h.get("num_tokens", 0)) for h in trainer.state.log_history)  # trainer-counted non-pad tokens
    out = {"gpu": torch.cuda.get_device_name(0), "steps": STEPS, "batch": BATCH, "max_len": MAXLEN,
           "train_seconds": round(dt, 1), "tokens": int(toks), "tokens_per_sec": round(toks / dt),
           "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
           "final_loss": trainer.state.log_history[-2].get("loss") if len(trainer.state.log_history) > 1 else None,
           "versions": {"trl": trl.__version__, "transformers": transformers.__version__, "torch": torch.__version__}}
    out = json.loads(json.dumps(out, default=str))  # plain types only: torch objects cannot unpickle locally
    print("METRICS " + json.dumps(out))
    return out


@app.local_entrypoint()
def main():
    print(train.remote())
