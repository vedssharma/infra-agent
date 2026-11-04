"""In-memory fake provider for tests and offline demos. Never touches a GPU or spends money.

It runs the real in-container runner (`gpu_agent.runner.run_job`) in a background thread against an
in-memory store, with a stand-in training command that prints the same event lines as train.py.
Pass `cmd=[sys.executable, "train/train.py"]` to run the real trainer locally instead.
"""

from __future__ import annotations

import copy
import subprocess
import sys
import threading
from collections.abc import Iterable, Sequence
from typing import Any

from gpu_agent.models import JobSpec
from gpu_agent.providers.base import CallPoll, CallStatus, StoreBackedProvider
from gpu_agent.runner import run_job

FAKE_TRAIN = r"""
import json, os, sys, time
P = "GPU_AGENT_EVENT "
spec = json.loads(os.environ["GPU_AGENT_SPEC"])
steps = spec.get("max_steps") or 3
delay = float(os.environ.get("FAKE_STEP_SECONDS", "0.01"))
def emit(**e): print(P + json.dumps(e), flush=True)
emit(event="start", device="FakeGPU", base_model=spec["base_model"])
if spec["base_model"] == "fail":
    print("Traceback: boom", flush=True); emit(event="error", error="boom"); sys.exit(1)
for s in range(1, steps + 1):
    time.sleep(delay)
    print(f"step {s}/{steps}", flush=True)
    emit(event="metrics", step=s, max_steps=steps, loss=round(2.0 - s * 0.01, 4), num_tokens=s * 1000)
emit(event="done", steps=steps, tokens=steps * 1000, final_loss=2.0 - steps * 0.01,
     hub_url=("https://huggingface.co/" + spec["hub_repo"]) if spec.get("hub_repo") else None)
"""


class MemoryStore:
    """dict-backed store with the modal.Dict subset the providers use.

    Values are deep-copied in and out, like modal.Dict's serialisation, so callers cannot share
    mutable state through it by accident.
    """

    def __init__(self) -> None:
        self.data: dict[Any, Any] = {}
        self.lock = threading.Lock()

    def get(self, key: Any, default: Any = None) -> Any:
        with self.lock:
            return copy.deepcopy(self.data.get(key, default))

    def put(self, key: Any, value: Any) -> bool:
        with self.lock:
            self.data[key] = copy.deepcopy(value)
        return True

    def keys(self) -> Iterable[Any]:
        with self.lock:
            return list(self.data)


class _Call:
    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen[str] | None = None
        self.result: dict[str, Any] | None = None
        self.cancelled = False
        self.timed_out = False
        self.timer: threading.Timer | None = None


class FakeProvider(StoreBackedProvider):
    name = "fake"
    price_as_of = "2026-11-04"
    usd_per_hour = {"FAKE-S": 0.10, "FAKE-L": 1.00}

    def __init__(self, store: MemoryStore | None = None, cmd: Sequence[str] | None = None):
        super().__init__(store or MemoryStore())
        self.cmd = list(cmd) if cmd else [sys.executable, "-c", FAKE_TRAIN]
        self.calls: dict[str, _Call] = {}

    def normalize_gpu(self, gpu: str) -> str:
        key = gpu.strip().upper()
        if key not in self.usd_per_hour:
            raise KeyError(f"unknown fake GPU {gpu!r}; known: {', '.join(self.usd_per_hour)}")
        return key

    def _spawn(self, spec: JobSpec, run_id: str, gpu: str, timeout_s: int) -> str:
        ref = f"fc-{run_id}"
        call = self.calls[ref] = _Call()

        def started(proc: subprocess.Popen[str]) -> None:
            call.proc = proc
            if call.cancelled or call.timed_out:  # stopped before the process existed
                proc.kill()

        def target() -> None:
            call.result = run_job(spec.to_train_json(), run_id, self.store, self.cmd,
                                  gpu_name=gpu, on_start=started)

        def ttl() -> None:  # provider-side TTL, like Modal's timeout=
            call.timed_out = True
            if call.proc:
                call.proc.kill()

        call.thread = threading.Thread(target=target, daemon=True)
        call.timer = threading.Timer(timeout_s, ttl)
        call.timer.daemon = True
        call.thread.start()
        call.timer.start()
        return ref

    def _poll(self, ref: str) -> CallPoll:
        call = self.calls.get(ref)
        if call is None:
            return CallPoll(CallStatus.FAILED, error="unknown call (fake provider restarted?)")
        if call.cancelled:
            return CallPoll(CallStatus.CANCELLED)
        if call.timed_out:
            return CallPoll(CallStatus.TIMED_OUT, error="provider timeout reached")
        if call.thread and call.thread.is_alive():
            return CallPoll(CallStatus.RUNNING)
        return CallPoll(CallStatus.SUCCEEDED, result=call.result)

    def _cancel(self, ref: str) -> None:
        call = self.calls.get(ref)
        if call is None:
            return
        call.cancelled = True
        if call.timer:
            call.timer.cancel()
        if call.proc:
            call.proc.kill()

    def wait(self, run_id: str, timeout: float = 30) -> None:
        """Test helper: block until the run's background thread finishes."""
        ref = self._meta(run_id)["provider_ref"]
        call = self.calls[ref]
        if call.thread:
            call.thread.join(timeout)
