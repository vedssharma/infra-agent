import sys
import threading

from gpu_agent import runner
from gpu_agent.events import EVENT_PREFIX
from gpu_agent.providers.fake import FAKE_TRAIN, MemoryStore
from tests.conftest import wait_for


def lines(store, run_id):
    out = []
    for i in range(store.get(runner.k_logn(run_id), 0)):
        out += store.get(runner.k_log(run_id, i))
    return out


def test_success_records_state_metrics_and_logs():
    store = MemoryStore()
    res = runner.run_job({"base_model": "m", "max_steps": 3, "hub_repo": "me/x"}, "r1", store,
                         [sys.executable, "-c", FAKE_TRAIN], gpu_name="A10")
    st = store.get(runner.k_state("r1"))
    assert res == st
    assert st["state"] == "succeeded" and st["exit_code"] == 0
    assert st["gpu_name"] == "FakeGPU"  # from the start event
    assert st["last_metrics"]["step"] == 3
    assert st["result"]["hub_url"] == "https://huggingface.co/me/x"
    assert st["finished_at"] >= st["started_at"]
    logs = lines(store, "r1")
    assert "step 2/3" in logs and any(line.startswith(EVENT_PREFIX) for line in logs)


def test_failure_keeps_traceback():
    store = MemoryStore()
    res = runner.run_job({"base_model": "fail"}, "r2", store, [sys.executable, "-c", FAKE_TRAIN])
    assert res["state"] == "failed" and res["error"] == "boom" and res["exit_code"] == 1
    assert "Traceback: boom" in lines(store, "r2")


def test_crash_without_events_is_failed():
    store = MemoryStore()
    res = runner.run_job({}, "r3", store, [sys.executable, "-c", "import sys; print('x'); sys.exit(3)"])
    assert res["state"] == "failed" and "code 3" in res["error"]


def test_success_exit_without_done_event_is_failed():
    store = MemoryStore()
    res = runner.run_job({}, "r4", store, [sys.executable, "-c", "print('hi')"])
    assert res["state"] == "failed"


def test_missing_command_is_failed_not_raised():
    store = MemoryStore()
    res = runner.run_job({}, "r5", store, ["/nonexistent/binary"])
    assert res["state"] == "failed" and "FileNotFoundError" in res["error"]


def test_logs_are_chunked(monkeypatch):
    monkeypatch.setattr(runner, "FLUSH_LINES", 10)
    store = MemoryStore()
    runner.run_job({}, "r6", store, [sys.executable, "-c", "for i in range(35): print(i)"])
    assert store.get(runner.k_logn("r6")) >= 4
    assert lines(store, "r6") == [str(i) for i in range(35)]


def test_cancelled_state_is_never_overwritten():
    """Once stop() marks a run cancelled, later writes from the dying runner must not revive it."""
    store = MemoryStore()
    code = (
        "import json, time\n"
        f"P = {EVENT_PREFIX!r}\n"
        "print(P + json.dumps({'event': 'start', 'device': 'X'}), flush=True)\n"
        "time.sleep(0.5)\n"
        "print(P + json.dumps({'event': 'done', 'steps': 1}), flush=True)\n"
    )
    t = threading.Thread(target=runner.run_job, args=({}, "r7", store, [sys.executable, "-c", code]))
    t.start()
    wait_for(lambda: (store.get(runner.k_state("r7")) or {}).get("gpu_name") == "X")
    store.put(runner.k_state("r7"), {"state": "cancelled", "error": "stopped by user"})
    t.join(10)
    st = store.get(runner.k_state("r7"))
    assert st["state"] == "cancelled" and st["error"] == "stopped by user"
