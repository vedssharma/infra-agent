# infra-agent (GPU Run Agent)

An MCP server that takes a plain-English GPU job request, picks the cheapest provider/GPU that fits (Modal, RunPod, Daytona), launches it, watches it, tears it down, and logs estimated vs. actual cost.

**Status:** M0 (provider spikes). See [PLAN.md](PLAN.md) and [docs/M0-spikes.md](docs/M0-spikes.md). Scope: [docs/GPU Run Agent — MVP Scope.pdf](docs/GPU%20Run%20Agent%20%E2%80%94%20MVP%20Scope.pdf).

## Spikes

Spike scripts in `spikes/` can spend real money. Each requires `--yes`, has a hard timeout, and tears down in a `finally`. Keys come from the environment (`MODAL_*`, `RUNPOD_API_KEY`, `DAYTONA_API_KEY`).

```bash
uv run python spikes/hf_config.py Qwen/Qwen2.5-1.5B   # free, no key
```
