# GPU Run Agent training image. One image for every provider (Modal, RunPod, Daytona).
#
#   docker build -t ghcr.io/vedssharma/gpu-agent-train:dev train/
#   docker run --gpus all -e HF_TOKEN -e GPU_AGENT_SPEC="$(cat examples/qwen05b-lora.json)" \
#       ghcr.io/vedssharma/gpu-agent-train:dev
#
# The job spec comes from $GPU_AGENT_SPEC (JSON) or `--spec path.json`. Progress is printed as
# JSON lines prefixed with "GPU_AGENT_EVENT " (see train.py).
#
# No ENTRYPOINT on purpose: Modal runs its own container command, and other providers pass the
# command explicitly. CMD makes `docker run` work as shown above.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false

COPY requirements.txt /opt/train/requirements.txt
RUN pip install -r /opt/train/requirements.txt

COPY train.py /opt/train/train.py

WORKDIR /opt/train
CMD ["python", "/opt/train/train.py"]
