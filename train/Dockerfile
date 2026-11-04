# GPU Run Agent training image. One image for every provider (Modal, RunPod, Daytona).
#
#   docker build -t ghcr.io/vedssharma/gpu-agent-train:dev train/
#   docker run --gpus all -e HF_TOKEN -e GPU_AGENT_SPEC="$(cat examples/qwen05b-lora.json)" \
#       ghcr.io/vedssharma/gpu-agent-train:dev
#
# The job spec comes from $GPU_AGENT_SPEC (JSON) or `--spec path.json`. Progress is printed as
# JSON lines prefixed with "GPU_AGENT_EVENT " (see train.py).
#
# Training libraries live in their own venv (/opt/venv), deliberately NOT on PATH. Modal's image
# builder (versions <= 2024.10) pip-installs its client's pinned deps into the system Python after
# these layers, which downgraded typing_extensions/aiohttp/... and broke `import torch` on the first
# real run. The system Python stays Modal's; train.py always runs with /opt/venv/bin/python.
#
# No ENTRYPOINT on purpose: Modal runs its own container command, and other providers pass the
# command explicitly. CMD makes `docker run` work as shown above.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_DATASETS_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false

# Triton JIT-compiles a small C launcher the first time a kernel runs (TRL's fused LM-head loss
# does this on CUDA), so the image needs a C compiler; python:3.12-slim ships without one.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/train/requirements.txt
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r /opt/train/requirements.txt

COPY train.py /opt/train/train.py

WORKDIR /opt/train
CMD ["/opt/venv/bin/python", "/opt/train/train.py"]
