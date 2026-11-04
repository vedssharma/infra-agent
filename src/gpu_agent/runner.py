"""In-container job runner: runs train.py, ships its logs and state to a key-value store.

Standard library only: this module runs inside the provider container (Modal imports it from the
mounted `gpu_agent` package). The store is any object with `get(key, default)` and `put(key, value)`
(a `modal.Dict` on Modal, a plain dict wrapper in tests).

Store layout for one run (all values plain JSON types):
    {run_id}/meta      written by the launcher: provider ref, gpu, timeout, created_at, spec
    {run_id}/state     written here: state, started_at, finished_at, last_metrics, result, error
    {run_id}/logn      number of log chunks written
    {run_id}/log/{i}   chunk i: a list of log lines
Log lines are batched into chunks so a chatty job does not make one store write per line.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from gpu_agent.events import parse_event

FLUSH_SECONDS = 2.0
FLUSH_LINES = 200
MAX_LINE_CHARS = 4000


class KVStore(Protocol):
    def get(self, key: Any, default: Any = None) -> Any: ...
    def put(self, key: Any, value: Any) -> Any: ...


def k_meta(run_id: str) -> str:
    return f"{run_id}/meta"


def k_state(run_id: str) -> str:
    return f"{run_id}/state"


def k_logn(run_id: str) -> str:
    return f"{run_id}/logn"


def k_log(run_id: str, i: int) -> str:
    return f"{run_id}/log/{i}"


class LogShipper:
    """Buffers lines and writes them to the store as numbered chunks (thread-safe)."""

    def __init__(self, store: KVStore, run_id: str):
        self.store, self.run_id = store, run_id
        self.buf: list[str] = []
        self.n = int(store.get(k_logn(run_id), 0) or 0)
        self.last_flush = time.monotonic()
        self.lock = threading.Lock()

    def add(self, line: str) -> None:
        with self.lock:
            self.buf.append(line[:MAX_LINE_CHARS])
            due = len(self.buf) >= FLUSH_LINES or time.monotonic() - self.last_flush >= FLUSH_SECONDS
        if due:
            self.flush()

    def flush(self) -> None:
        with self.lock:
            if not self.buf:
                self.last_flush = time.monotonic()
                return
            chunk, self.buf = self.buf, []
            self.store.put(k_log(self.run_id, self.n), chunk)
            self.n += 1
            self.store.put(k_logn(self.run_id), self.n)
            self.last_flush = time.monotonic()


def _put_state(store: KVStore, run_id: str, state: dict[str, Any]) -> None:
    """Write state, but never undo a stop: once the launcher marks a run cancelled, it stays so."""
    cur = store.get(k_state(run_id)) or {}
    if cur.get("state") == "cancelled":
        state["state"] = "cancelled"
        state["error"] = cur.get("error") or state.get("error")
    store.put(k_state(run_id), state)


def _now() -> float:
    return round(time.time(), 3)


def run_job(spec: dict[str, Any], run_id: str, store: KVStore, cmd: Sequence[str],
            gpu_name: str | None = None, env: dict[str, str] | None = None,
            on_start: Callable[[subprocess.Popen[str]], None] | None = None) -> dict[str, Any]:
    """Run `cmd` (train.py) with the spec, stream its output to the store, return a plain summary dict.

    Never raises for a training failure: the failure is recorded in the store and returned, so the
    provider sees a clean function result with an error field (M0 lesson: return plain types).
    """
    state: dict[str, Any] = {"state": "running", "started_at": _now(), "finished_at": None,
                             "gpu_name": gpu_name, "last_metrics": None, "result": None, "error": None}
    store.put(k_state(run_id), state)
    shipper = LogShipper(store, run_id)

    stop_ticker = threading.Event()

    def ticker() -> None:  # flush quiet periods too (model download prints little)
        while not stop_ticker.wait(FLUSH_SECONDS):
            shipper.flush()

    threading.Thread(target=ticker, daemon=True).start()
    try:
        proc_env = {**os.environ, **(env or {}), "GPU_AGENT_SPEC": json.dumps(spec),
                    "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=proc_env)
        if on_start:
            on_start(proc)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            shipper.add(line)
            ev = parse_event(line)
            if ev is None:
                continue
            kind = ev.get("event")
            if kind == "start":
                state["gpu_name"] = ev.get("device") or state["gpu_name"]
            elif kind == "metrics":
                state["last_metrics"] = ev
            elif kind == "done":
                state["result"] = ev
            elif kind == "error":
                state["error"] = ev.get("error")
            else:
                continue
            _put_state(store, run_id, state)
        code = proc.wait()
        if code != 0 and not state["error"]:
            state["error"] = f"train process exited with code {code}"
        state["state"] = "succeeded" if code == 0 and state["result"] is not None else "failed"
        state["exit_code"] = code
    except Exception:
        state["state"] = "failed"
        state["error"] = traceback.format_exc()
    finally:
        stop_ticker.set()
        state["finished_at"] = _now()
        try:
            shipper.flush()
        finally:
            _put_state(store, run_id, state)
    return json.loads(json.dumps(state, default=str))
