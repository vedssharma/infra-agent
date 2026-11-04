"""M0 spike (UNTESTED, needs DAYTONA_API_KEY): create a GPU sandbox, run nvidia-smi, always delete.

    uv run --extra spikes python spikes/daytona_spike.py --yes
Docs say GPU sandboxes use the `daytona-gpu` snapshot; GPU-type selection params are unverified.
"""
import sys

from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams, Daytona, GpuType, Resources

if "--yes" not in sys.argv:
    sys.exit("refusing to spend money without --yes")

d = Daytona()
if "--rtx4090" in sys.argv:  # request a specific cheaper GPU via a CUDA image
    params = CreateSandboxFromImageParams(
        image="nvidia/cuda:12.4.1-base-ubuntu22.04", resources=Resources(gpu=1, gpu_type=GpuType.RTX_4090),
        ttl_minutes=5, auto_stop_interval=5, auto_delete_interval=0)
else:
    params = CreateSandboxFromSnapshotParams(snapshot="daytona-gpu", ttl_minutes=5, auto_stop_interval=5, auto_delete_interval=0)
sb = d.create(params, timeout=300)
try:
    print(sb.process.exec("nvidia-smi").result)
finally:
    d.delete(sb)
    print("deleted", sb.id)
