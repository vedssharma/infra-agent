"""Provider interface (quote/launch/status/logs/stop/list_active) and the shared store-backed base.

Every provider in this project runs the same training image and the same in-container runner
(`gpu_agent.runner`), which writes logs and state to a key-value store the launcher can read. So the
provider-specific part is small: start a call, poll it, cancel it. `StoreBackedProvider` implements
the rest once.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from gpu_agent.models import JobSpec, LogPage, Quote, RunInfo, RunState
from gpu_agent.runner import k_log, k_logn, k_meta, k_state

MAX_TIMEOUT_S = 24 * 3600


class ProviderError(RuntimeError):
    """A provider call failed in a way the user should see (bad auth, capacity, bad GPU name...)."""


class RunNotFound(ProviderError):
    pass


class Provider(ABC):
    """What the CLI and (from M5) the MCP server use. One adapter per provider."""

    name: str

    @abstractmethod
    def gpus(self) -> list[str]: ...

    @abstractmethod
    def quote(self, spec: JobSpec, gpu: str, timeout_s: int) -> Quote: ...

    @abstractmethod
    def launch(self, spec: JobSpec, gpu: str, timeout_s: int) -> RunInfo:
        """Start the job and return immediately. `timeout_s` is enforced provider-side."""

    @abstractmethod
    def status(self, run_id: str) -> RunInfo: ...

    @abstractmethod
    def logs(self, run_id: str, cursor: int = 0) -> LogPage: ...

    @abstractmethod
    def stop(self, run_id: str) -> RunInfo: ...

    @abstractmethod
    def list_active(self) -> list[RunInfo]: ...


class Store(Protocol):
    def get(self, key: Any, default: Any = None) -> Any: ...
    def put(self, key: Any, value: Any) -> Any: ...
    def keys(self) -> Iterable[Any]: ...


class CallStatus(StrEnum):
    RUNNING = "running"  # queued or executing; the store tells which
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class CallPoll:
    status: CallStatus
    error: str | None = None
    result: dict[str, Any] | None = field(default=None)


def new_run_id(provider: str) -> str:
    return f"{provider}-{uuid.uuid4().hex[:12]}"


def provider_of(run_id: str) -> str:
    return run_id.split("-", 1)[0]


def _dt(ts: float | None) -> datetime | None:
    return datetime.fromtimestamp(ts, UTC) if ts else None


class StoreBackedProvider(Provider):
    """Shared logic for providers whose container runs `gpu_agent.runner.run_job` against a store."""

    price_as_of: str
    usd_per_hour: dict[str, float]
    min_timeout_s = 60

    def __init__(self, store: Store):
        self.store = store

    # -- provider-specific hooks ------------------------------------------------------------
    @abstractmethod
    def normalize_gpu(self, gpu: str) -> str: ...

    @abstractmethod
    def _spawn(self, spec: JobSpec, run_id: str, gpu: str, timeout_s: int) -> str:
        """Start the call; return the provider-native reference used by _poll/_cancel."""

    @abstractmethod
    def _poll(self, ref: str) -> CallPoll: ...

    @abstractmethod
    def _cancel(self, ref: str) -> None: ...

    # -- shared implementation --------------------------------------------------------------
    def gpus(self) -> list[str]:
        return list(self.usd_per_hour)

    def quote(self, spec: JobSpec, gpu: str, timeout_s: int) -> Quote:
        g = self._gpu(gpu)
        return Quote(provider=self.name, gpu=g, usd_per_hour=self.usd_per_hour[g],
                     timeout_s=self._check_timeout(timeout_s), price_as_of=self.price_as_of)

    def launch(self, spec: JobSpec, gpu: str, timeout_s: int) -> RunInfo:
        g = self._gpu(gpu)
        timeout_s = self._check_timeout(timeout_s)
        run_id = new_run_id(self.name)
        meta = {"run_id": run_id, "provider": self.name, "gpu": g, "timeout_s": timeout_s,
                "usd_per_hour": self.usd_per_hour[g], "created_at": round(time.time(), 3),
                "spec": spec.to_train_json(), "provider_ref": None}
        self.store.put(k_meta(run_id), meta)  # before spawning, so a crash still leaves a trace
        meta["provider_ref"] = self._spawn(spec, run_id, g, timeout_s)
        self.store.put(k_meta(run_id), meta)
        return self.status(run_id)

    def status(self, run_id: str) -> RunInfo:
        meta = self._meta(run_id)
        st: dict[str, Any] | None = self.store.get(k_state(run_id))
        ref = meta.get("provider_ref")
        poll = self._poll(ref) if ref else CallPoll(CallStatus.FAILED, error="launch did not complete")

        stored = RunState(st["state"]) if st else None
        killed = {CallStatus.CANCELLED: RunState.CANCELLED, CallStatus.TIMED_OUT: RunState.TIMED_OUT}
        if poll.status in killed and stored not in (RunState.SUCCEEDED, killed[poll.status]):
            # A kill by the provider (TTL) or by stop() beats whatever the dying process wrote.
            state = killed[poll.status]
            st = {**(st or {}), "state": state.value,
                  "finished_at": (st or {}).get("finished_at") or round(time.time(), 3),
                  "error": poll.error or (st or {}).get("error") or f"run {state.value}"}
            self.store.put(k_state(run_id), st)
        elif stored is not None and stored.terminal:
            state = stored
        elif poll.status is CallStatus.RUNNING:
            state = RunState.RUNNING if st else RunState.PENDING
        else:
            # The call ended but the container never wrote a terminal state (killed by TTL or
            # cancel, or died). Record it so later reads are stable and the cost stops growing.
            state = {CallStatus.SUCCEEDED: RunState.FAILED, CallStatus.FAILED: RunState.FAILED,
                     CallStatus.CANCELLED: RunState.CANCELLED,
                     CallStatus.TIMED_OUT: RunState.TIMED_OUT}[poll.status]
            if poll.status is CallStatus.SUCCEEDED and poll.result and poll.result.get("state"):
                st = poll.result  # runner returned its final state but the store write was lost
                state = RunState(st["state"])
            else:
                st = {**(st or {}), "state": state.value, "finished_at": round(time.time(), 3),
                      "error": (st or {}).get("error") or poll.error or f"run {state.value}"}
            self.store.put(k_state(run_id), st)
        return self._info(meta, st, state)

    def logs(self, run_id: str, cursor: int = 0) -> LogPage:
        info = self.status(run_id)  # read state first: a terminal state is written after the last flush
        n = int(self.store.get(k_logn(run_id), 0) or 0)
        lines: list[str] = []
        for i in range(max(cursor, 0), n):
            lines.extend(self.store.get(k_log(run_id, i)) or [])
        new_cursor = max(cursor, n)
        return LogPage(lines=lines, cursor=new_cursor, done=info.state.terminal and new_cursor >= n)

    def stop(self, run_id: str) -> RunInfo:
        meta = self._meta(run_id)
        info = self.status(run_id)
        if info.state.terminal:
            return info
        if meta.get("provider_ref"):
            self._cancel(meta["provider_ref"])
        st = self.store.get(k_state(run_id)) or {}
        self.store.put(k_state(run_id), {**st, "state": RunState.CANCELLED.value,
                                         "finished_at": round(time.time(), 3),
                                         "error": st.get("error") or "stopped by user"})
        return self.status(run_id)

    def list_active(self) -> list[RunInfo]:
        out = []
        for key in list(self.store.keys()):
            if isinstance(key, str) and key.endswith("/meta"):
                info = self.status(key[: -len("/meta")])
                if not info.state.terminal:
                    out.append(info)
        return sorted(out, key=lambda r: r.created_at)

    # -- helpers ----------------------------------------------------------------------------
    def _gpu(self, gpu: str) -> str:
        try:
            return self.normalize_gpu(gpu)
        except KeyError as e:
            raise ProviderError(str(e.args[0])) from None

    def _check_timeout(self, timeout_s: int) -> int:
        if not self.min_timeout_s <= int(timeout_s) <= MAX_TIMEOUT_S:
            raise ProviderError(f"timeout must be between {self.min_timeout_s} s and {MAX_TIMEOUT_S} s, "
                                f"got {timeout_s}")
        return int(timeout_s)

    def _meta(self, run_id: str) -> dict[str, Any]:
        meta = self.store.get(k_meta(run_id))
        if not meta:
            raise RunNotFound(f"no run {run_id!r} on {self.name}")
        return meta

    def _info(self, meta: dict[str, Any], st: dict[str, Any] | None, state: RunState) -> RunInfo:
        st = st or {}
        started, finished = st.get("started_at"), st.get("finished_at")
        rate = meta.get("usd_per_hour")
        cost = None
        if started and rate is not None:
            end = finished if state.terminal and finished else time.time()
            cost = round(max(end - started, 0) * rate / 3600, 4)
        result = st.get("result")
        return RunInfo(
            run_id=meta["run_id"], provider=meta["provider"], gpu=meta["gpu"], state=state,
            timeout_s=meta["timeout_s"], created_at=_dt(meta["created_at"]),  # type: ignore[arg-type]
            started_at=_dt(started), finished_at=_dt(finished) if state.terminal else None,
            usd_per_hour=rate, est_cost_usd=cost, last_metrics=st.get("last_metrics"),
            result=result, error=st.get("error"),
            hub_url=(result or {}).get("hub_url"), provider_ref=meta.get("provider_ref"),
            device=st.get("gpu_name"),
        )

