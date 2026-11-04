"""M0 spike (UNTESTED, needs Modal auth): run nvidia-smi on a T4 with a hard timeout.

    uv run --extra spikes modal run spikes/modal_spike.py
Cost: a few cents. Timeout is enforced by Modal itself (timeout=120).
"""
import modal

app = modal.App("gpu-agent-spike")


@app.function(gpu="T4", timeout=120)
def smi() -> str:
    import subprocess

    return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout


@app.local_entrypoint()
def main():
    print(smi.remote())
