# M0 spike status (2026-11-04)

All M0 spikes were run on 2026-11-04. **M0 is complete**; total cloud spend was well under $1 of the $50 dev budget (estimated, not read from bills).

## Results

| Provider | Verified | Price (list) | TTL / teardown mechanism |
|---|---|---|---|
| **Modal** | T4 `nvidia-smi`; A10 training run. `modal run` stops the app on completion. | T4 $0.59/hr, L4 $0.80, A10 $1.10, L40S $1.95, A100-40 $2.10, A100-80 $2.50, H100 $3.95, H200 $4.54, B200 $6.25 (per-second billing; pricing page, no API; $30/mo free on Starter) | `timeout=` on the function (native) |
| **RunPod** | GraphQL `gpuTypes` pricing; REST create/status/delete; capacity fallback list. Community cloud was sometimes sold out ("no instances currently available") and then SECURE worked; community RTX-class pod cost $0.22/hr, secure A100-80 PCIe $1.19/hr. | Queryable via API (community and secure $/hr; `securePrice: 0` seems to mean not offered) | **No native TTL.** In-container self-termination works: the pod ran `runpodctl remove pod $RUNPOD_POD_ID` after 40s and was gone at 44s. Outside DELETE returns 204. REST `cloudType` is only `SECURE` or `COMMUNITY`. |
| **Daytona** | `daytona-gpu` snapshot gives an H100 80GB; image + `Resources(gpu=1, gpu_type=GpuType.RTX_4090)` gives an RTX 4090 (23 GiB) from `nvidia/cuda` image. | H100 $3.95/hr on-demand ($2.27 preemptible), RTX 4090 $0.99 ($0.57 preemptible) per Daytona pricing page via search, plus small CPU/RAM/disk charges; **verify on dashboard** | **Native `ttl_minutes`** (wall-clock since creation), plus `auto_stop_interval` and `auto_delete_interval`; `spot=True` option. GPU types: H100, H200, B200, B300, RTX-PRO-6000, RTX-4090, RTX-5090, MI355X. |

Notes:
- 5 older STOPPED Daytona sandboxes exist in the account that these spikes did not create.
- No leaked pods, sandboxes or Modal apps from the spikes.

## Training smoke run (Qwen2.5-0.5B, LoRA r=16, TRL 1.17.1, transformers 5.22.1, torch 2.15)

20 steps, batch 4, max_len 512, bf16, Modal A10, Capybara subset:

| Metric | Value |
|---|---|
| Train time | 9.2 s (plus image build and model download not counted) |
| Tokens processed | 32,884 (trainer-counted, non-pad) |
| Throughput | ~3,570 tokens/s |
| Peak VRAM allocated / reserved | 2.36 GiB / 2.75 GiB |
| Loss | 1.89 to 1.77 |

First estimator check: weights 0.92 GiB + fp32 logits up to ~1.2 GiB (4 x 512 x 151,936 x 4 B) + LoRA/optimizer/activations/overhead lands in the right range of the measured 2.36 GiB. The estimator should model logits explicitly.

## Lessons for M1
- Return plain JSON types from Modal functions; torch objects fail to unpickle in a local env without torch. Wrap remote code and return tracebacks as strings.
- In transformers 5, `apply_chat_template(tokenize=True)` returns a dict; don't count `len()` of it. Use the trainer's `num_tokens` log.
- The same GPU name can differ slightly across runs (A10 vs A10G); key throughput tables by Modal GPU string.
- Pin versions in the training image; unpinned installs pulled very new TRL/transformers.
- Quotes must handle sold-out capacity and fall through to the next option.
- Hub config lookup (`spikes/hf_config.py`) gives accurate param counts; Qwen's 151,936-token vocab makes the logits tensor large.
