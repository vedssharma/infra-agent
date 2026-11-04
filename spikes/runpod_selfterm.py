"""M0 spike: can a RunPod pod terminate itself from inside the container? (TTL mechanism)

    uv run --env-file .env python spikes/runpod_selfterm.py --yes
Pod sleeps 40s then tries runpodctl, then a REST DELETE with the pod's injected key. We poll until the pod
disappears (expected ~1-2 min) and always DELETE from outside as a safety net.
"""
import os
import sys
import time

import httpx

H = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}
REST = "https://rest.runpod.io/v1"
CMD = ('sleep 40; echo selfterm-start; '
       'runpodctl remove pod $RUNPOD_POD_ID || runpodctl pod delete $RUNPOD_POD_ID || '
       'curl -sS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID; '
       'sleep 300')
GPUS = ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A4000", "NVIDIA L4",
        "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40", "NVIDIA L40S", "NVIDIA A100 80GB PCIe"]

if "--yes" not in sys.argv:
    sys.exit("refusing to spend money without --yes")

pod_id = None
for cloud in ("COMMUNITY", "SECURE"):
    r = httpx.post(f"{REST}/pods", headers=H, timeout=60, json={
        "name": "gpu-agent-selfterm", "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "gpuTypeIds": GPUS, "gpuTypePriority": "availability", "gpuCount": 1, "cloudType": cloud,
        "containerDiskInGb": 20, "dockerStartCmd": ["bash", "-c", CMD]})
    print(cloud, r.status_code)
    if r.status_code == 201:
        pod_id = r.json()["id"]
        print("pod", pod_id, "costPerHr", r.json().get("costPerHr"))
        break
if not pod_id:
    sys.exit("could not create pod")
t0 = time.time()
try:
    gone = None
    while time.time() - t0 < 180:
        g = httpx.get(f"{REST}/pods/{pod_id}", headers=H, timeout=30)
        if g.status_code == 404:
            gone = time.time() - t0
            break
        time.sleep(10)
    print("SELF-TERMINATED after %.0fs" % gone if gone else "NOT self-terminated within 180s")
finally:
    d = httpx.delete(f"{REST}/pods/{pod_id}", headers=H, timeout=30)
    print("safety delete:", d.status_code)
