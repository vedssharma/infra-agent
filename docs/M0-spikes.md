# M0 spike status (2026-11-03)

No provider API keys were available in the dev environment, so provider scripts in `spikes/` are written but **untested**. Findings below are from docs/web search only, not from running anything.

## Done
- **Hub config lookup** — `spikes/hf_config.py` works with no key. Param estimates match the real models (Qwen2.5-0.5B ≈ 0.49B, 1.5B ≈ 1.54B). Qwen vocab is 151,936, so the fp32 logits tensor is ~2.3 GiB at 4096 tokens per step: the estimator must model it.

## Findings (unverified until a key run)
- **Daytona** docs list GPU sandboxes (H100, H200, RTX Pro 6000, RTX 4090, RTX 5090), one GPU per sandbox, ephemeral, created from the `daytona-gpu` snapshot; some third-party comparisons say the default auto-stop can kill long GPU jobs, so set `auto_stop_interval` explicitly. Other sources claim no GPU support; they appear stale. Sources: <https://www.daytona.io/changelog/nvidia-gpu-runners-and-dashboard-terminal-access>, <https://www.daytona.io/changelog/gpu-type-selector-and-windows-sandbox-class>.
- **RunPod**: REST `terminate` action and `runpodctl pod delete` exist. A `--terminate-after` flag was claimed in search results but is **not** in the runpodctl pod reference page, so plan on in-container self-termination. Pod listing exists (`list_active` maps to it). Pricing API (GraphQL `gpuTypes`) not yet confirmed.
- **Modal**: `timeout=` on the function is the TTL; `modal app logs` / `modal app stop` exist per third-party sources. Official price list still to capture from modal.com/pricing.

## Remaining (needs keys and a few dollars)
- [ ] Run `modal_spike.py`, `runpod_spike.py --yes`, `daytona_spike.py --yes`; record GPU types, prices, startup latency.
- [ ] 20-step TRL LoRA smoke run on Qwen2.5-0.5B (`train/` does not exist yet; first task of M1 prep).
- [ ] Decide RunPod TTL mechanism; decide Modal log streaming approach.
