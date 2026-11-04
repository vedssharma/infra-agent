"""M0 spike (UNTESTED, needs RUNPOD_API_KEY): list GPU prices; optionally create + terminate one pod.

    uv run python spikes/runpod_spike.py                 # prices only, free
    uv run python spikes/runpod_spike.py --yes           # also launches one pod for <=2 min, always terminates
API shapes are from RunPod docs and must be verified on first run.
"""
import os
import sys
import time

import httpx

KEY = os.environ["RUNPOD_API_KEY"]
H = {"Authorization": f"Bearer {KEY}"}
GQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"


def prices() -> None:
    q = "{ gpuTypes { id displayName memoryInGb securePrice communityPrice } }"
    r = httpx.post(GQL, json={"query": q}, headers=H, timeout=30)
    r.raise_for_status()
    for g in sorted(r.json()["data"]["gpuTypes"], key=lambda g: g.get("communityPrice") or 99):
        print(g)


def pod_roundtrip(cloud: str) -> None:
    body = {"name": "gpu-agent-spike", "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "gpuTypeIds": ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A4000",
                           "NVIDIA L4", "NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40", "NVIDIA L40S",
                           "NVIDIA GeForce RTX 5090", "NVIDIA RTX 6000 Ada Generation", "NVIDIA A100 80GB PCIe",
                           "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe"], "gpuTypePriority": "availability", "gpuCount": 1,
            "cloudType": cloud,
            "containerDiskInGb": 20, "dockerStartCmd": ["sleep", "120"]}
    r = httpx.post(f"{REST}/pods", json=body, headers=H, timeout=60)
    print(r.status_code, r.text[:500])
    r.raise_for_status()
    pod_id = r.json()["id"]
    try:
        for _ in range(12):
            s = httpx.get(f"{REST}/pods/{pod_id}", headers=H, timeout=30).json()
            print(s.get("desiredStatus"), s.get("costPerHr"))
            time.sleep(10)
    finally:
        d = httpx.delete(f"{REST}/pods/{pod_id}", headers=H, timeout=30)
        print("terminate:", d.status_code)


if __name__ == "__main__":
    prices()
    if "--yes" in sys.argv:
        for cloud in ("COMMUNITY", "SECURE"):
            try:
                pod_roundtrip(cloud)
                break
            except httpx.HTTPStatusError as e:
                print(f"{cloud} failed: {e.response.status_code}")
