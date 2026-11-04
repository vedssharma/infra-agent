"""M0 spike (UNTESTED, needs DAYTONA_API_KEY): create a GPU sandbox, run nvidia-smi, always delete.

    uv run --extra spikes python spikes/daytona_spike.py --yes
Docs say GPU sandboxes use the `daytona-gpu` snapshot; GPU-type selection params are unverified.
"""
import sys

from daytona import CreateSandboxFromSnapshotParams, Daytona

if "--yes" not in sys.argv:
    sys.exit("refusing to spend money without --yes")

d = Daytona()
sb = d.create(CreateSandboxFromSnapshotParams(snapshot="daytona-gpu", auto_stop_interval=5, auto_delete_interval=0))
try:
    print(sb.process.exec("nvidia-smi").result)
finally:
    d.delete(sb)
    print("deleted", sb.id)
