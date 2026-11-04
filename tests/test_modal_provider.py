"""Modal adapter against a stub of the SDK surface it uses (the real exception classes are kept)."""

from __future__ import annotations

import modal
import pytest

from gpu_agent.models import JobSpec, RunState
from gpu_agent.providers import modal_app
from gpu_agent.providers.base import ProviderError
from gpu_agent.providers.fake import MemoryStore
from gpu_agent.providers.modal import ModalProvider
from gpu_agent.runner import k_state


class StubCall:
    def __init__(self, object_id: str, outcome):
        self.object_id = object_id
        self.outcome = outcome  # an exception instance to raise, or a result to return
        self.cancelled_with = None

    def get(self, timeout=None):
        assert timeout == 0, "status polls must never block"
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def cancel(self, terminate_containers=False):
        self.cancelled_with = terminate_containers
        self.outcome = modal.exception.InputCancellation()


class StubModal:
    exception = modal.exception

    def __init__(self):
        self.calls: dict[str, StubCall] = {}
        self.spawned: list[dict] = []
        self.next_outcome = TimeoutError()  # what modal 1.6 really raises while a call is unfinished
        self.spawn_error: Exception | None = None
        stub = self

        class Function:
            @staticmethod
            def from_name(app_name, fn_name):
                assert (app_name, fn_name) == (modal_app.APP_NAME, modal_app.FUNCTION_NAME)
                return Function()

            def with_options(self, **opts):
                self.opts = opts
                return self

            def spawn(self, spec, run_id):
                if stub.spawn_error:
                    raise stub.spawn_error
                call = StubCall(f"fc-{len(stub.calls)}", stub.next_outcome)
                stub.calls[call.object_id] = call
                stub.spawned.append({"opts": self.opts, "spec": spec, "run_id": run_id})
                return call

        class FunctionCall:
            @staticmethod
            def from_id(ref):
                return stub.calls[ref]

        self.Function = Function
        self.FunctionCall = FunctionCall


@pytest.fixture
def stub():
    return StubModal()


@pytest.fixture
def p(stub):
    return ModalProvider(store=MemoryStore(), modal_module=stub, deploy=False)


def test_launch_sets_gpu_ttl_and_spec(p, stub):
    info = p.launch(JobSpec(max_steps=10), "a10g", 900)
    assert info.state is RunState.PENDING  # spawned, container has not reported yet
    (s,) = stub.spawned
    assert s["opts"] == {"gpu": "A10", "timeout": 900}
    assert s["spec"]["max_steps"] == 10 and s["run_id"] == info.run_id
    assert info.run_id.startswith("modal-") and info.provider_ref == "fc-0"
    assert info.usd_per_hour == 1.10


def test_running_when_container_reported(p):
    info = p.launch(JobSpec(), "A10", 900)
    p.store.put(k_state(info.run_id), {"state": "running", "started_at": 1.0, "gpu_name": "NVIDIA A10"})
    st = p.status(info.run_id)
    assert st.state is RunState.RUNNING and st.device == "NVIDIA A10" and st.est_cost_usd > 0


@pytest.mark.parametrize(("exc", "state"), [
    (modal.exception.FunctionTimeoutError("t"), RunState.TIMED_OUT),
    (modal.exception.InputCancellation(), RunState.CANCELLED),
    (modal.exception.OutputExpiredError(), RunState.FAILED),
    (modal.exception.RemoteError("container died"), RunState.FAILED),
    (modal.exception.ExecutionError("bad result"), RunState.FAILED),
])
def test_kills_and_crashes_override_stale_running_state(p, stub, exc, state):
    info = p.launch(JobSpec(), "A10", 900)
    p.store.put(k_state(info.run_id), {"state": "running", "started_at": 1.0})
    stub.calls[info.provider_ref].outcome = exc
    st = p.status(info.run_id)
    assert st.state is state and st.finished_at is not None and st.error
    assert p.store.get(k_state(info.run_id))["state"] == state.value  # persisted


def test_unfinished_call_is_pending_not_failed(p, stub):
    """Regression: modal raises the builtin TimeoutError (not modal.exception.TimeoutError) on get(timeout=0)."""
    info = p.launch(JobSpec(), "A10", 900)
    assert isinstance(stub.calls[info.provider_ref].outcome, TimeoutError)
    assert not isinstance(stub.calls[info.provider_ref].outcome, modal.exception.TimeoutError)
    assert p.status(info.run_id).state is RunState.PENDING
    assert p.store.get(k_state(info.run_id)) is None  # nothing persisted


def test_unexpected_poll_error_is_not_recorded_as_failure(p, stub):
    info = p.launch(JobSpec(), "A10", 900)
    p.store.put(k_state(info.run_id), {"state": "running", "started_at": 1.0})
    stub.calls[info.provider_ref].outcome = ConnectionError("network down")
    with pytest.raises(ProviderError, match="network down"):
        p.status(info.run_id)
    assert p.store.get(k_state(info.run_id))["state"] == "running"


def test_success_uses_runner_state(p, stub):
    info = p.launch(JobSpec(hub_repo="me/a"), "A10", 900)
    final = {"state": "succeeded", "started_at": 1.0, "finished_at": 61.0,
             "result": {"event": "done", "hub_url": "https://huggingface.co/me/a", "steps": 5}}
    p.store.put(k_state(info.run_id), final)
    stub.calls[info.provider_ref].outcome = final
    st = p.status(info.run_id)
    assert st.state is RunState.SUCCEEDED and st.hub_url == "https://huggingface.co/me/a"
    assert st.est_cost_usd == pytest.approx(60 * 1.10 / 3600, abs=1e-4)


def test_result_used_if_store_write_was_lost(p, stub):
    info = p.launch(JobSpec(), "A10", 900)
    stub.calls[info.provider_ref].outcome = {"state": "succeeded", "started_at": 1.0, "finished_at": 2.0,
                                             "result": {"steps": 1}}
    assert p.status(info.run_id).state is RunState.SUCCEEDED


def test_stop_terminates_container(p, stub):
    info = p.launch(JobSpec(), "A10", 900)
    p.store.put(k_state(info.run_id), {"state": "running", "started_at": 1.0})
    st = p.stop(info.run_id)
    assert stub.calls[info.provider_ref].cancelled_with is True
    assert st.state is RunState.CANCELLED


def test_list_active(p, stub):
    a = p.launch(JobSpec(), "A10", 900)
    b = p.launch(JobSpec(), "T4", 900)
    stub.calls[b.provider_ref].outcome = modal.exception.InputCancellation()
    assert [r.run_id for r in p.list_active()] == [a.run_id]


def test_launch_errors_become_provider_errors(p, stub):
    stub.spawn_error = modal.exception.AuthError("token missing")
    with pytest.raises(ProviderError, match="AuthError: token missing"):
        p.launch(JobSpec(), "A10", 900)
    with pytest.raises(ProviderError, match="unknown Modal GPU"):
        p.launch(JobSpec(), "RTX-9090", 900)


def test_modal_app_definition():
    assert (modal_app.TRAIN_DIR / "Dockerfile").is_file()
    assert (modal_app.TRAIN_DIR / "train.py").is_file()
    assert isinstance(modal_app.app, modal.App)
    assert isinstance(modal_app.run_job, modal.Function)


def test_training_requirements_are_pinned():
    reqs = [ln.strip() for ln in (modal_app.TRAIN_DIR / "requirements.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    assert reqs and all("==" in r for r in reqs), reqs
    names = {r.split("==")[0] for r in reqs}
    assert {"torch", "transformers", "trl", "peft", "datasets", "huggingface-hub"} <= names


def test_container_side_modules_import_only_stdlib_and_modal():
    """modal_app and runner run in the training image, which has no pydantic or typer."""
    import ast
    import sys
    from pathlib import Path

    import gpu_agent

    pkg = Path(gpu_agent.__file__).parent
    allowed = set(sys.stdlib_module_names) | {"modal", "__future__"}
    for rel in ("__init__.py", "runner.py", "events.py", "providers/__init__.py", "providers/modal_app.py"):
        for node in ast.parse((pkg / rel).read_text()).body:  # module level only; lazy imports are fine
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for m in mods:
                assert m.split(".")[0] in allowed or m == "gpu_agent.events", f"{rel} imports {m} at module level"


def test_deploys_once_per_process(stub, monkeypatch):
    import contextlib

    deployed: list[str] = []
    monkeypatch.setattr(modal_app.app, "deploy", lambda name: deployed.append(name))
    stub.enable_output = contextlib.nullcontext
    p = ModalProvider(store=MemoryStore(), modal_module=stub, deploy=True)
    p.launch(JobSpec(), "A10", 900)
    p.launch(JobSpec(), "A10", 900)
    assert deployed == [modal_app.APP_NAME]
