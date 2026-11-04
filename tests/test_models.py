import pytest
from pydantic import ValidationError

from gpu_agent.events import EVENT_PREFIX, parse_event
from gpu_agent.models import JobSpec, Quote, RunState


def test_jobspec_matches_train_py_defaults(train_module):
    """The CLI-side spec and the image-side trainer must agree on every key and default."""
    assert JobSpec().to_train_json() == train_module.DEFAULTS


def test_event_prefix_matches_train_py(train_module):
    assert train_module.EVENT_PREFIX == EVENT_PREFIX


def test_jobspec_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValidationError):
        JobSpec(learning_rat=1e-4)  # typo
    with pytest.raises(ValidationError):
        JobSpec(max_steps=0)
    with pytest.raises(ValidationError):
        JobSpec(hub_repo="no-namespace")
    assert JobSpec(hub_repo="me/my-adapter").hub_repo == "me/my-adapter"


def test_quote_worst_case_cost():
    q = Quote(provider="modal", gpu="A10", usd_per_hour=1.10, timeout_s=1800, price_as_of="2026-11-04")
    assert q.max_cost_usd == pytest.approx(0.55)


def test_terminal_states():
    assert not RunState.PENDING.terminal and not RunState.RUNNING.terminal
    assert all(s.terminal for s in (RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED, RunState.TIMED_OUT))


def test_parse_event():
    assert parse_event(EVENT_PREFIX + '{"event": "metrics", "step": 3}') == {"event": "metrics", "step": 3}
    assert parse_event("plain log line") is None
    assert parse_event(EVENT_PREFIX + "{not json") is None
    assert parse_event(EVENT_PREFIX + '{"no_event": 1}') is None


def test_train_py_load_spec(train_module, tmp_path, monkeypatch):
    p = tmp_path / "spec.json"
    p.write_text('{"max_steps": 7}')
    spec = train_module.load_spec(["--spec", str(p)])
    assert spec["max_steps"] == 7 and spec["base_model"] == "Qwen/Qwen2.5-0.5B"
    monkeypatch.setenv("GPU_AGENT_SPEC", '{"bogus": 1}')
    with pytest.raises(ValueError, match="bogus"):
        train_module.load_spec([])
    monkeypatch.delenv("GPU_AGENT_SPEC")
    with pytest.raises(SystemExit):
        train_module.load_spec([])


def test_train_py_reports_errors_as_events(train_module, monkeypatch, capsys):
    monkeypatch.setenv("GPU_AGENT_SPEC", '{"bogus": 1}')
    assert train_module.main([]) == 1
    ev = parse_event(capsys.readouterr().out.strip().splitlines()[-1])
    assert ev and ev["event"] == "error" and "bogus" in ev["error"]
