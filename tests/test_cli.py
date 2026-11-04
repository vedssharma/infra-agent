import json

import pytest
from typer.testing import CliRunner

from gpu_agent import cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def fresh_providers():
    cli._providers.clear()
    yield
    cli._providers.clear()


def test_parse_duration():
    assert cli.parse_duration("90s") == 90
    assert cli.parse_duration("30m") == 1800
    assert cli.parse_duration("2h") == 7200
    assert cli.parse_duration("120") == 120


def test_quote_lists_modal_prices():
    r = runner.invoke(cli.app, ["quote", "--gpu", "A10", "--timeout", "1h"])
    assert r.exit_code == 0, r.output
    assert "A10" in r.output and "$ 1.10/h" in r.output and "worst case $  1.10" in r.output


def test_launch_requires_yes():
    r = runner.invoke(cli.app, ["launch", "--provider", "fake", "--gpu", "FAKE-S", "--timeout", "5m"])
    assert r.exit_code == 1 and "--yes" in r.output
    assert cli._providers["fake"].list_active() == []  # nothing launched


def test_launch_refuses_over_max_cost():
    r = runner.invoke(cli.app, ["launch", "--gpu", "B200", "--timeout", "2h", "--yes"])
    assert r.exit_code == 1 and "above --max-cost" in r.output


def test_launch_rejects_bad_spec(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"learning_rat": 1}))
    r = runner.invoke(cli.app, ["launch", "--spec", str(f), "--provider", "fake", "--gpu", "FAKE-S", "--yes"])
    assert r.exit_code != 0 and "learning_rat" in r.output


def test_launch_follow_end_to_end(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"max_steps": 3, "hub_repo": "me/adapter"}))
    r = runner.invoke(cli.app, ["launch", "--spec", str(f), "--max-steps", "5", "--provider", "fake",
                                "--gpu", "FAKE-S", "--timeout", "5m", "--yes", "--follow"])
    assert r.exit_code == 0, r.output
    assert "step 5/5" in r.output  # flag overrides the spec file
    assert "succeeded" in r.output and "https://huggingface.co/me/adapter" in r.output

    run_id = next(w for w in r.output.split() if w.startswith("fake-"))
    s = runner.invoke(cli.app, ["status", run_id, "--json"])
    assert s.exit_code == 0 and json.loads(s.output)["state"] == "succeeded"
    lg = runner.invoke(cli.app, ["logs", run_id])
    assert "step 1/5" in lg.output


def test_failed_run_exits_nonzero():
    r = runner.invoke(cli.app, ["launch", "--model", "fail", "--provider", "fake", "--gpu", "FAKE-S",
                                "--timeout", "5m", "--yes", "--follow"])
    assert r.exit_code == 1 and "failed" in r.output and "boom" in r.output


def test_unknown_run():
    r = runner.invoke(cli.app, ["status", "fake-nope"])
    assert r.exit_code == 1 and "no run" in r.output
