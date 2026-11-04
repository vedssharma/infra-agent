"""Typed job specs and run records shared by the CLI, providers and (later) the MCP server."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JobSpec(BaseModel):
    """What to train. Field names and defaults mirror `train/train.py` DEFAULTS (a test checks it)."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = "Qwen/Qwen2.5-0.5B"
    dataset: str = "trl-lib/Capybara"
    dataset_split: str = "train"
    dataset_limit: int | None = Field(default=None, ge=1, description="use only the first N examples")
    max_steps: int | None = Field(default=None, ge=1, description="overrides epochs when set")
    epochs: float = Field(default=1.0, gt=0)
    per_device_batch_size: int = Field(default=4, ge=1)
    grad_accum: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    max_length: int = Field(default=512, ge=16)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: str | list[str] = "all-linear"
    gradient_checkpointing: bool = False
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=50, ge=1)
    seed: int = 42
    hub_repo: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$", description="namespace/name")
    hub_private: bool = True
    output_dir: str = "/tmp/gpu-agent-out"

    def to_train_json(self) -> dict[str, Any]:
        """The dict handed to train.py (GPU_AGENT_SPEC)."""
        return self.model_dump(mode="json")


class RunState(StrEnum):
    PENDING = "pending"      # accepted by the provider; container not started yet (queue, image build)
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"  # provider-side TTL fired

    @property
    def terminal(self) -> bool:
        return self not in (RunState.PENDING, RunState.RUNNING)


class Quote(BaseModel):
    provider: str
    gpu: str
    usd_per_hour: float
    timeout_s: int
    max_cost_usd: float = Field(default=0.0, description="worst case: the whole TTL billed at list price; "
                                "computed from rate and timeout when omitted")
    price_as_of: str

    @model_validator(mode="before")
    @classmethod
    def _fill_max_cost(cls, data: Any) -> Any:
        if isinstance(data, dict) and "max_cost_usd" not in data:
            data = {**data, "max_cost_usd": round(data["usd_per_hour"] * data["timeout_s"] / 3600, 4)}
        return data


class RunInfo(BaseModel):
    run_id: str
    provider: str
    gpu: str
    device: str | None = Field(default=None, description="GPU name reported inside the container")
    state: RunState
    timeout_s: int
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    usd_per_hour: float | None = None
    est_cost_usd: float | None = Field(default=None, description="container seconds x list price; not a bill")
    last_metrics: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    hub_url: str | None = None
    provider_ref: str | None = Field(default=None, description="provider-native id, e.g. Modal function call id")


class LogPage(BaseModel):
    lines: list[str]
    cursor: int = Field(description="pass back to get the next page")
    done: bool = Field(description="run is terminal and every line has been returned")
