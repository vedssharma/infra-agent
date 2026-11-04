"""Provider contract, exercised through the fake (which runs the real in-container runner)."""

import time

import pytest

from gpu_agent.models import JobSpec, RunState
from gpu_agent.providers.base import ProviderError, RunNotFound, provider_of
from gpu_agent.providers.fake import FakeProvider
from tests.conftest import wait_for


@pytest.fixture
def p():
    return FakeProvider()


def test_quote(p):
    q = p.quote(JobSpec(), "fake-l", 3600)
    assert (q.gpu, q.usd_per_hour, q.max_cost_usd) == ("FAKE-L", 1.0, 1.0)
    with pytest.raises(ProviderError, match="unknown fake GPU"):
        p.quote(JobSpec(), "H100", 3600)
    with pytest.raises(ProviderError, match="timeout"):
        p.quote(JobSpec(), "FAKE-S", 5)


def test_launch_runs_to_success(p):
    info = p.launch(JobSpec(max_steps=4, hub_repo="me/adapter"), "FAKE-S", 600)
    assert provider_of(info.run_id) == "fake"
    assert info.state in (RunState.PENDING, RunState.RUNNING)
    p.wait(info.run_id)
    done = p.status(info.run_id)
    assert done.state is RunState.SUCCEEDED
    assert done.device == "FakeGPU"
    assert done.hub_url == "https://huggingface.co/me/adapter"
    assert done.result and done.result["steps"] == 4
    assert done.est_cost_usd is not None and done.est_cost_usd >= 0
    assert done.finished_at is not None
    assert p.list_active() == []


def test_logs_paging_with_cursor(p, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_SECONDS", "0.3")
    info = p.launch(JobSpec(max_steps=8), "FAKE-S", 600)
    seen, cursor, pages = [], 0, 0
    while True:
        page = p.logs(info.run_id, cursor)
        seen += page.lines
        assert page.cursor >= cursor
        cursor = page.cursor
        pages += 1
        if page.done:
            break
        time.sleep(0.2)
    assert pages > 1  # streamed while running, not only at the end
    assert [ln for ln in seen if ln.startswith("step ")] == [f"step {i}/8" for i in range(1, 9)]
    again = p.logs(info.run_id, cursor)
    assert again.lines == [] and again.done


def test_failed_run(p):
    info = p.launch(JobSpec(base_model="fail"), "FAKE-S", 600)
    p.wait(info.run_id)
    st = p.status(info.run_id)
    assert st.state is RunState.FAILED and st.error == "boom"


def test_stop_cancels_and_stays_cancelled(p, monkeypatch):
    monkeypatch.setenv("FAKE_STEP_SECONDS", "0.5")
    info = p.launch(JobSpec(max_steps=100), "FAKE-S", 600)
    wait_for(lambda: p.status(info.run_id).state is RunState.RUNNING)
    assert [r.run_id for r in p.list_active()] == [info.run_id]
    stopped = p.stop(info.run_id)
    assert stopped.state is RunState.CANCELLED
    p.wait(info.run_id)
    final = p.status(info.run_id)
    assert final.state is RunState.CANCELLED and final.finished_at is not None
    assert final.error == "stopped by user"
    assert p.logs(info.run_id).done
    assert p.stop(info.run_id).state is RunState.CANCELLED  # idempotent


def test_provider_ttl_kills_run(monkeypatch):
    p = FakeProvider()
    p.min_timeout_s = 1
    monkeypatch.setenv("FAKE_STEP_SECONDS", "0.5")
    info = p.launch(JobSpec(max_steps=100), "FAKE-S", 1)
    p.wait(info.run_id)
    st = p.status(info.run_id)
    assert st.state is RunState.TIMED_OUT and "timeout" in (st.error or "")
    cost_then = st.est_cost_usd
    time.sleep(0.2)
    assert p.status(info.run_id).est_cost_usd == cost_then  # cost frozen once terminal


def test_unknown_run(p):
    with pytest.raises(RunNotFound):
        p.status("fake-doesnotexist")
