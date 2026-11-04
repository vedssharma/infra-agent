"""`gpu-agent` command line: quote, launch, status, logs, stop, list.

Shares the provider library with the MCP server that lands in M5. Launching spends real money, so
`launch` prints the worst-case cost (TTL x list price), refuses anything above --max-cost, and
needs --yes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from gpu_agent.models import JobSpec, RunInfo
from gpu_agent.providers import PROVIDERS, get_provider
from gpu_agent.providers.base import Provider, ProviderError, provider_of

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Launch, watch and stop GPU fine-tuning jobs (M1: Modal).")

DEFAULT_MAX_COST = 5.0
_providers: dict[str, Provider] = {}


def provider(name: str) -> Provider:
    """One provider instance per process (the fake one keeps its runs in memory)."""
    if name not in PROVIDERS:
        raise typer.BadParameter(f"unknown provider {name!r}; known: {', '.join(PROVIDERS)}")
    if name not in _providers:
        _providers[name] = get_provider(name)
    return _providers[name]


def for_run(run_id: str) -> Provider:
    return provider(provider_of(run_id))


def fail(msg: str) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def parse_duration(text: str) -> int:
    """'90s', '30m', '2h' or plain seconds -> seconds."""
    t = text.strip().lower()
    mult = {"s": 1, "m": 60, "h": 3600}.get(t[-1:], None)
    try:
        return int(float(t[:-1]) * mult) if mult else int(t)
    except ValueError:
        raise typer.BadParameter(f"bad duration {text!r}; use e.g. 90s, 30m, 2h") from None


def build_spec(spec_file: Path | None, overrides: dict[str, Any]) -> JobSpec:
    base: dict[str, Any] = json.loads(spec_file.read_text()) if spec_file else {}
    base.update({k: v for k, v in overrides.items() if v is not None})
    try:
        return JobSpec(**base)
    except Exception as e:
        raise typer.BadParameter(str(e)) from None


def show(info: RunInfo, as_json: bool = False) -> None:
    if as_json:
        typer.echo(info.model_dump_json(indent=2))
        return
    color = {"succeeded": typer.colors.GREEN, "failed": typer.colors.RED,
             "timed_out": typer.colors.RED, "cancelled": typer.colors.YELLOW}.get(info.state.value)
    typer.echo(f"run      {info.run_id}")
    typer.echo("state    " + typer.style(info.state.value, fg=color, bold=True))
    typer.echo(f"gpu      {info.gpu}" + (f" ({info.device})" if info.device else ""))
    typer.echo(f"ttl      {info.timeout_s // 60} min")
    if info.started_at:
        end = info.finished_at.timestamp() if info.finished_at else time.time()
        typer.echo(f"runtime  {end - info.started_at.timestamp():.0f} s")
    if info.est_cost_usd is not None:
        typer.echo(f"cost     ~${info.est_cost_usd:.4f} (container time x ${info.usd_per_hour}/h list price)")
    m = info.last_metrics
    if m and not info.result:
        typer.echo(f"progress step {m.get('step')}/{m.get('max_steps')}  loss {m.get('loss')}  "
                   f"{m.get('tokens_per_sec')} tok/s")
    if info.result:
        r = info.result
        typer.echo(f"result   {r.get('steps')} steps, loss {r.get('first_loss')} -> {r.get('final_loss')}, "
                   f"{r.get('tokens_per_sec')} tok/s, peak VRAM {r.get('peak_vram_gib')} GiB")
    if info.hub_url:
        typer.echo(f"hub      {info.hub_url}")
    if info.error:
        typer.secho("error    " + info.error.strip().splitlines()[-1], fg=typer.colors.RED)


def follow_logs(p: Provider, run_id: str, cursor: int = 0, poll_s: float = 3.0) -> int:
    while True:
        page = p.logs(run_id, cursor)
        for line in page.lines:
            typer.echo(line)
        cursor = page.cursor
        if page.done:
            return cursor
        time.sleep(poll_s)


@app.command()
def quote(
    gpu: Annotated[list[str] | None, typer.Option(help="GPU(s) to quote; default: all")] = None,
    timeout: Annotated[str, typer.Option(help="TTL, e.g. 30m")] = "30m",
    provider_name: Annotated[str, typer.Option("--provider")] = "modal",
) -> None:
    """List prices and the worst-case cost (whole TTL billed) per GPU."""
    p = provider(provider_name)
    secs = parse_duration(timeout)
    try:
        for g in gpu or p.gpus():
            q = p.quote(JobSpec(), g, secs)
            typer.echo(f"{q.provider:6} {q.gpu:10} ${q.usd_per_hour:5.2f}/h  worst case ${q.max_cost_usd:6.2f}"
                       f"  (prices as of {q.price_as_of})")
    except ProviderError as e:
        fail(str(e))


@app.command()
def launch(
    spec_file: Annotated[Path | None, typer.Option("--spec", exists=True, dir_okay=False,
                                                   help="JSON job spec; flags below override it")] = None,
    model: Annotated[str | None, typer.Option(help="base model on the Hub")] = None,
    dataset: Annotated[str | None, typer.Option(help="Hub dataset (or local .jsonl)")] = None,
    max_steps: Annotated[int | None, typer.Option()] = None,
    hub_repo: Annotated[str | None, typer.Option(help="push the adapter to namespace/name")] = None,
    gpu: Annotated[str, typer.Option()] = "A10",
    timeout: Annotated[str, typer.Option(help="provider-enforced TTL, e.g. 30m")] = "30m",
    max_cost: Annotated[float, typer.Option(help="refuse if worst-case cost exceeds this (USD)")]
    = DEFAULT_MAX_COST,
    provider_name: Annotated[str, typer.Option("--provider")] = "modal",
    yes: Annotated[bool, typer.Option("--yes", help="confirm spending money")] = False,
    follow: Annotated[bool, typer.Option("--follow", help="stream logs until the run ends")] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Launch a LoRA fine-tune. Returns at once unless --follow."""
    spec = build_spec(spec_file, {"base_model": model, "dataset": dataset, "max_steps": max_steps,
                                  "hub_repo": hub_repo})
    p = provider(provider_name)
    try:
        q = p.quote(spec, gpu, parse_duration(timeout))
    except ProviderError as e:
        fail(str(e))
        return
    typer.echo(f"{spec.base_model} on {q.provider} {q.gpu} at ${q.usd_per_hour}/h, TTL {q.timeout_s // 60} min: "
               f"worst case ${q.max_cost_usd:.2f}")
    if q.max_cost_usd > max_cost:
        fail(f"worst case ${q.max_cost_usd:.2f} is above --max-cost ${max_cost:.2f}; shorten --timeout or "
             "raise --max-cost")
    if not spec.hub_repo:
        typer.secho("note: no --hub-repo, the adapter is discarded when the container exits",
                    fg=typer.colors.YELLOW)
    if not yes:
        fail("launching spends money; re-run with --yes")
    try:
        info = p.launch(spec, q.gpu, q.timeout_s)
    except ProviderError as e:
        fail(str(e))
        return
    if follow:
        follow_logs(p, info.run_id)
        info = p.status(info.run_id)
    show(info, as_json)
    if not follow and not as_json:
        typer.echo(f"\nnext: gpu-agent logs {info.run_id} --follow")
    if info.state.terminal and info.state.value != "succeeded":
        raise typer.Exit(1)


@app.command()
def status(run_id: str, as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Show a run's state, progress, cost so far and result."""
    try:
        show(for_run(run_id).status(run_id), as_json)
    except ProviderError as e:
        fail(str(e))


@app.command()
def logs(
    run_id: str,
    cursor: Annotated[int, typer.Option(help="resume from a cursor printed by a previous call")] = 0,
    follow: Annotated[bool, typer.Option("--follow", "-f")] = False,
) -> None:
    """Print a run's logs (polling; Ctrl-C stops following, not the run)."""
    p = for_run(run_id)
    try:
        if follow:
            follow_logs(p, run_id, cursor)
        else:
            page = p.logs(run_id, cursor)
            for line in page.lines:
                typer.echo(line)
            typer.secho(f"[cursor {page.cursor}{', done' if page.done else ''}]", err=True, dim=True)
    except ProviderError as e:
        fail(str(e))


@app.command()
def stop(run_id: str) -> None:
    """Cancel a run and terminate its container."""
    try:
        show(for_run(run_id).stop(run_id))
    except ProviderError as e:
        fail(str(e))


@app.command("list")
def list_runs(provider_name: Annotated[str, typer.Option("--provider")] = "modal") -> None:
    """List runs that are still pending or running."""
    runs = provider(provider_name).list_active()
    if not runs:
        typer.echo("no active runs")
    for r in runs:
        cost = f"~${r.est_cost_usd:.4f}" if r.est_cost_usd is not None else "-"
        typer.echo(f"{r.run_id}  {r.state.value:8} {r.gpu:10} {cost}  created {r.created_at:%Y-%m-%d %H:%M}Z")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
