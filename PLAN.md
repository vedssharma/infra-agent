# GPU Run Agent — Plan and Tech Stack

Companion to `docs/GPU Run Agent — MVP Scope.pdf`. The scope says *what*; this says *how* and *in what order*.

## Goal

A local MCP server that takes a plain-English GPU job request, picks the cheapest provider/GPU that can run it, launches it, watches it, tears it down, and logs estimated vs. actual cost. Headline demo from Claude Code: *"Fine-tune Qwen2.5-1.5B with LoRA on this dataset, keep it under $5."*

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language/tooling | Python 3.12, `uv`, `ruff`, `pyright`, `pytest` | All provider SDKs are Python |
| MCP server | Official `mcp` SDK (FastMCP), stdio | Claude Code spawns it as a subprocess |
| Specs/config | Pydantic v2 | Strict typed tool I/O and limits file |
| Ledger | stdlib `sqlite3`, WAL, hand-written migrations | Tiny schema; no ORM |
| CLI | Typer, sharing the core library with the server | M1–M3 need a CLI before the server exists |
| Secrets | `keyring`, env var fallback | Keys never enter model context |
| HTTP | `httpx` | Pricing APIs, Hub `config.json` lookups |
| Providers | `modal`, `runpod` (REST/GraphQL via httpx), `daytona` SDK; one adapter each | Interface: `quote/launch/status/logs/stop/list_active` |
| Training image | One Docker image: PyTorch, transformers, TRL, PEFT, bitsandbytes, datasets, huggingface_hub | Same image on all providers; pushed to GHCR |
| Storage | Hugging Face Hub (`hub_strategy="checkpoint"`) | Neutral layer; providers stay interchangeable |
| Tests | `respx`/`vcrpy` recorded responses + in-memory fake provider | CI never touches a GPU |

## Repo layout

```
src/gpu_agent/   models, estimator, pricing/, ledger, guardrails, watchdog,
                 providers/{base,modal,runpod,daytona,fake}, server, cli
train/           Dockerfile, train.py (image entrypoint)
spikes/          M0 throwaway scripts
tests/  bench/  docs/
```

## Design decisions beyond the scope doc

1. **Approval is a human step.** `launch_run` confirms via MCP elicitation (fallback: `gpu-agent approve <plan_id>`). The ledger stores a hash of the plan spec; launch refuses on mismatch.
2. **Limits file is agent-proof.** `~/.config/gpu-agent/limits.toml`, chmod 444, plus a Claude Code deny rule on the path. Server only reads it.
3. **Teardown never depends on the laptop.** Provider-side TTL is the real guarantee: Modal `timeout=`; RunPod needs in-container self-termination (and/or `--terminate-after` if confirmed); Daytona auto-stop/auto-delete. The watchdog runs as its own process (launchd/`nohup`), not inside the MCP server.
4. **No blocking tools.** `launch_run` returns immediately; logs are polled via `get_logs(run_id, cursor)`.
5. **`train.py` emits JSON-line metrics** (loss, step, tokens/s, `torch.cuda.max_memory_allocated`) that feed `get_run` and estimator calibration.
6. **Estimator v1.** VRAM = weights + LoRA params + optimizer/grads + activations (batch × seq × hidden × layers, reduced by grad checkpointing) + logits (batch × seq × vocab × 4 B; dominant for Qwen's ~150k vocab) + ~1.5 GB CUDA overhead, × 1.2 margin. Shape from Hub `config.json`, no weight download. Runtime = tokens × epochs ÷ measured tokens/s per (GPU, model size).
7. **Cost is "estimated actual".** Billed seconds × rate at teardown; reconcile against provider bills later.

## Milestones

Each ends in something runnable; move on when it works. Hard **$50** total dev cloud budget.

- **M0 — Spikes. DONE (2026-11-04).** Provider unknowns answered; see `docs/M0-spikes.md`.
- **M1 — Modal end to end. DONE (2026-11-04).** Training image + `train.py`, provider base interface, Modal adapter, CLI launch/status/logs/stop. Done: 0.5B LoRA run pushes an adapter to the Hub.
- **M2 — Ledger and guardrails.** SQLite ledger, limits file, per-run budget/TTL, watchdog with orphan sweep, Hub checkpointing. Done: a run is killed at its cap; a deliberately leaked job is swept.
- **M3 — RunPod and Daytona adapters.** Shared contract test suite (fake + recorded). Dated pricing catalog with stale flags.
- **M4 — Estimator and `get_quotes`.** Formula, safety margin, OOM auto-retry on next GPU size, throughput table seeded from M1–M3 runs.
- **M5 — MCP server.** Eight tools, approval gate, typed error taxonomy, wire into Claude Code, run the headline demo.
- **M6 — Benchmarks and write-up.** ~10-job suite vs. naive A100 baseline; README, demo video, blog post.

Stretch (after benchmarks): spot + auto-resume, learned estimator, read-only dashboard, fourth provider.

## M1 exit criteria

- [x] Training image: `train/Dockerfile`, pinned `train/requirements.txt`, `train/train.py` with JSON-line metrics and Hub push.
- [x] Provider interface (`providers/base.py`) and a fake provider running the real in-container runner.
- [x] Modal adapter: deploy, spawn with per-run GPU/TTL, status, logs, stop, list. Log streaming: in-container runner writes chunks to a `modal.Dict` (cursor = chunk index).
- [x] CLI: `quote`, `launch`, `status`, `logs`, `stop`, `list`.
- [x] Tests: contract via fake, Modal adapter against an SDK stub, CLI, and real CPU LoRA training through the full stack.
- [x] 0.5B LoRA run on Modal pushes an adapter to the Hub (`examples/qwen05b-lora.json`, A10): private repo `vedssharma/gpu-agent-qwen2.5-0.5b-lora-m1`. First deploy with image build took ~100 s. Cold start and cost per run are not recorded yet; capture them from `gpu-agent status` on the next run.

Lessons from the first real runs (each fixed and covered by a test):
- `FunctionCall.get(timeout=0)` raises the builtin `TimeoutError` while a call is unfinished, not `modal.exception.TimeoutError` (#2).
- Modal image builder <= 2024.10 pip-installs its client's pinned deps into the system Python after the Dockerfile, which downgraded `typing_extensions` and broke `import torch`. Training runs from `/opt/venv`, which is kept off PATH (#3).
- TRL's fused LM-head loss JIT-compiles a Triton kernel on CUDA, so the image needs `gcc` (#4). M0 didn't hit this because Modal's `debian_slim` has a compiler.

## M0 exit criteria

- [x] Daytona GPU sandbox created via SDK; `nvidia-smi` works; GPU types and prices known; native `ttl_minutes` found. *Fallback if unusable: Lambda or Vast.ai.*
- [x] RunPod: pod create/list/terminate via REST; GPU price query works; in-container self-termination verified as the TTL mechanism.
- [x] Modal: GPU function with `timeout=` runs; price list captured. (Log streaming approach still to choose in M1.)
- [x] Hub `config.json` shape lookup works (`spikes/hf_config.py`).
- [x] 20-step TRL LoRA smoke run on Qwen2.5-0.5B records peak VRAM (2.36 GiB) and tokens/s (~3,570 on A10). See `docs/M0-spikes.md`.

## Risks

| Risk | Mitigation |
|---|---|
| Daytona GPU support is new/limited | M0 spike; fall back to another provider |
| VRAM estimate too low | 1.2× margin, one auto-retry on next GPU size, log every miss |
| Prices not in APIs | Dated hand-maintained catalog; flag stale quotes |
| Provider billing lag | Report "estimated actual", reconcile later |
| Cold start / image pull dominates small jobs | Measure in M1; include in quotes |
| HF token lives in containers | Fine-grained write token scoped to one namespace, separate from main token |
| Dev cost creep | $50 cap enforced by the project's own global guardrail |
