# infra-agent (GPU Run Agent)

An MCP server that takes a plain-English GPU job request, picks the cheapest provider/GPU that fits (Modal, RunPod, Daytona), launches it, watches it, tears it down, and logs estimated vs. actual cost.

**Status:** M0 complete; M1 (Modal end to end) complete; M2 (ledger and guardrails) next. See [PLAN.md](PLAN.md) and [docs/M0-spikes.md](docs/M0-spikes.md). Scope: [docs/GPU Run Agent — MVP Scope.pdf](docs/GPU%20Run%20Agent%20%E2%80%94%20MVP%20Scope.pdf).

## Quick start (M1: Modal)

One-time setup:

```bash
uv sync
uv run modal token new                                     # Modal auth (~/.modal.toml)
uv run modal secret create gpu-agent-hf HF_TOKEN=hf_...    # fine-grained token, write access to one namespace
```

Then:

```bash
uv run gpu-agent quote --timeout 30m                       # list prices and worst-case cost per GPU
uv run gpu-agent launch --spec examples/qwen05b-lora.json --gpu A10 --timeout 30m --yes
uv run gpu-agent logs  <run_id> --follow                   # Ctrl-C stops following, not the run
uv run gpu-agent status <run_id>
uv run gpu-agent stop  <run_id>
uv run gpu-agent list                                      # runs still pending or running
```

`launch` returns at once (add `--follow` to stream logs), prints the worst-case cost (whole TTL at list price), refuses above `--max-cost` (default $5), and needs `--yes`. The TTL is Modal's own function `timeout`, so a run dies on time even if this machine is off. The first launch builds the training image on Modal (several minutes, CPU-billed); later launches reuse it.

### How it fits together

- `train/` holds the provider-neutral training image: `Dockerfile`, pinned `requirements.txt`, and `train.py` (TRL + PEFT LoRA SFT). `train.py` takes a JSON job spec and prints `GPU_AGENT_EVENT {json}` lines (start, metrics, done, error) with loss, tokens/s and peak VRAM. With `hub_repo` set it creates the repo first (bad tokens fail before GPU time), pushes checkpoints (`hub_strategy="checkpoint"`) and the final adapter.
- `gpu_agent.providers.base` is the provider interface: `quote`, `launch`, `status`, `logs`, `stop`, `list_active`.
- `gpu_agent.runner` runs inside the container, runs `train.py`, and writes state and batched log chunks to a key-value store. On Modal that store is the `gpu-agent-runs` `modal.Dict`; status combines it with the `FunctionCall` result to catch TTL kills and cancels.
- `gpu_agent.providers.modal` deploys `modal_app` (image built from `train/Dockerfile`), sets GPU and TTL per launch with `Function.with_options`, and `spawn`s.
- `gpu_agent.providers.fake` runs the same runner locally in a thread for tests.
- Cost shown is "estimated actual": container seconds times list price, not a bill.

## Development

```bash
uv run pytest -q              # fast suite: no GPU, no cloud
uv run ruff check src tests train && uv run pyright
# Real LoRA training on CPU with a tiny random model (needs the training libraries):
uv venv /tmp/trainenv && VIRTUAL_ENV=/tmp/trainenv uv pip install -r train/requirements.txt
GPU_AGENT_TRAIN_PYTHON=/tmp/trainenv/bin/python uv run pytest -q tests/test_train_cpu.py
```

## Spikes

Spike scripts in `spikes/` can spend real money. Each requires `--yes`, has a hard timeout, and tears down in a `finally`. Keys come from the environment (`MODAL_*`, `RUNPOD_API_KEY`, `DAYTONA_API_KEY`).

```bash
uv run python spikes/hf_config.py Qwen/Qwen2.5-1.5B   # free, no key
```
