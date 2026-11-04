# M0 spike status (2026-11-04)

Keys were added on 2026-11-04 and the spikes were run. Results are in **Run results** below; the **Findings** section further down is from docs/web search before the runs.

## Run results (2026-11-04)
| Provider | Result |
|---|---|
| Modal | **Works.** `modal run` on a T4 (15 GiB) returned `nvidia-smi` output; app stopped itself on completion. |
| RunPod | **Works.** GraphQL `gpuTypes` pricing works (community and secure $/hr, e.g. RTX 4090 $0.34 community, A100 80GB PCIe $1.19, H100 SXM $2.69); `securePrice: 0` appears to mean "not offered". After adding funds, REST pod create succeeded on the first fallback list in `SECURE` ($1.19/hr, reported via `costPerHr`), status polling showed RUNNING for ~2 min, and `DELETE /pods/{id}` returned 204; listing afterwards showed no pods. Community cloud returned "no instances currently available" for the same list, so quotes must handle sold-out capacity. REST `cloudType` accepts only `SECURE` or `COMMUNITY`. Still unverified: in-container self-termination as the TTL mechanism. |
| Daytona | **Works.** `daytona-gpu` snapshot gave a single **H100 80GB**; `nvidia-smi` ran; sandbox deleted. Note: the snapshot gave no GPU-type choice in my script, so the default is an expensive H100. Price/hour not yet found, and must be before it can be quoted. 5 older STOPPED sandboxes exist in the account (not created by these spikes). |

## Done
- **Hub config lookup** — `spikes/hf_config.py` works with no key. Param estimates match the real models (Qwen2.5-0.5B ≈ 0.49B, 1.5B ≈ 1.54B). Qwen vocab is 151,936, so the fp32 logits tensor is ~2.3 GiB at 4096 tokens per step: the estimator must model it.

## Findings (unverified until a key run)
- **Daytona** docs list GPU sandboxes (H100, H200, RTX Pro 6000, RTX 4090, RTX 5090), one GPU per sandbox, ephemeral, created from the `daytona-gpu` snapshot; some third-party comparisons say the default auto-stop can kill long GPU jobs, so set `auto_stop_interval` explicitly. Other sources claim no GPU support; they appear stale. Sources: <https://www.daytona.io/changelog/nvidia-gpu-runners-and-dashboard-terminal-access>, <https://www.daytona.io/changelog/gpu-type-selector-and-windows-sandbox-class>.
- **RunPod**: REST `terminate` action and `runpodctl pod delete` exist. A `--terminate-after` flag was claimed in search results but is **not** in the runpodctl pod reference page, so plan on in-container self-termination. Pod listing exists (`list_active` maps to it). Pricing API (GraphQL `gpuTypes`) not yet confirmed.
- **Modal**: `timeout=` on the function is the TTL; `modal app logs` / `modal app stop` exist per third-party sources. Official price list still to capture from modal.com/pricing.

## Remaining
- [ ] Verify RunPod TTL: pod self-terminating from inside the container (create/status/terminate are done).
- [ ] Find Daytona GPU pricing and how to pick a GPU type (changelog mentions a type selector); capture startup latency for all three.
- [ ] Capture Modal's price list from its pricing page.
- [ ] 20-step TRL LoRA smoke run on Qwen2.5-0.5B (`train/` does not exist yet; first task of M1 prep).
- [ ] Decide RunPod TTL mechanism; decide Modal log streaming approach.
