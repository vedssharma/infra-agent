"""Real training, on CPU, through the whole stack: FakeProvider -> runner -> train/train.py -> store.

Uses a tiny randomly initialised Qwen2 model and a word-level tokenizer built on the fly, so it needs
no network. Needs the training libraries (torch, transformers, trl, peft, datasets); set
GPU_AGENT_TRAIN_PYTHON to an interpreter that has them, e.g. a venv made from train/requirements.txt.
Skipped when none is available. Takes about a minute.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_agent.events import parse_event
from gpu_agent.models import JobSpec, RunState
from gpu_agent.providers.fake import FakeProvider
from tests.conftest import TRAIN_PY

pytestmark = pytest.mark.slow

MAKE_TINY_MODEL = r"""
import json, sys
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

out, data = Path(sys.argv[1]), Path(sys.argv[2])
words = sorted({w for line in data.read_text().splitlines() for w in json.loads(line)["text"].split()})
vocab = {t: i for i, t in enumerate(["[UNK]", "[PAD]", "[EOS]", *words])}
tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
fast.save_pretrained(out)
cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                  pad_token_id=1, eos_token_id=2, bos_token_id=2, tie_word_embeddings=True)
Qwen2ForCausalLM(cfg).save_pretrained(out)
"""

NEEDED = "import torch, transformers, trl, peft, datasets, tokenizers"


def train_python() -> str | None:
    for cand in (os.environ.get("GPU_AGENT_TRAIN_PYTHON"), sys.executable):
        if cand and subprocess.run([cand, "-c", NEEDED], capture_output=True).returncode == 0:
            return cand
    return None


@pytest.fixture(scope="module")
def py() -> str:
    found = train_python()
    if not found:
        pytest.skip("training libraries not installed; set GPU_AGENT_TRAIN_PYTHON")
    return found


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, py) -> tuple[Path, Path]:
    d = tmp_path_factory.mktemp("tiny")
    data = d / "data.jsonl"
    sentences = ["the cat sat on the mat", "a dog ran in the park", "the sun is warm today",
                 "we train a small model", "loss goes down slowly", "gpus cost money per second"]
    data.write_text("\n".join(json.dumps({"text": sentences[i % len(sentences)]}) for i in range(48)) + "\n")
    model = d / "model"
    subprocess.run([py, "-c", MAKE_TINY_MODEL, str(model), str(data)], check=True,
                   env={**os.environ, "HF_HUB_OFFLINE": "1"})
    return model, data


def offline_env() -> dict[str, str]:
    return {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_lora_run_end_to_end_on_cpu(py, tiny, tmp_path, monkeypatch):
    for k, v in offline_env().items():
        monkeypatch.setenv(k, v)
    model, data = tiny
    out = tmp_path / "out"
    p = FakeProvider(cmd=[py, str(TRAIN_PY)])
    spec = JobSpec(base_model=str(model), dataset=str(data), max_steps=6, per_device_batch_size=4,
                   max_length=32, logging_steps=2, learning_rate=1e-3, output_dir=str(out))
    info = p.launch(spec, "FAKE-S", 600)
    p.wait(info.run_id, timeout=300)
    st = p.status(info.run_id)
    logs = p.logs(info.run_id).lines
    assert st.state is RunState.SUCCEEDED, (st.error, "\n".join(logs[-40:]))

    events = [e for e in map(parse_event, logs) if e]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    metrics = [e for e in events if e["event"] == "metrics"]
    assert [m["step"] for m in metrics] == [2, 4, 6]
    assert all(m["loss"] is not None and m["num_tokens"] > 0 for m in metrics)

    r = st.result
    assert r and r["steps"] == 6 and r["tokens"] > 0 and r["final_loss"] is not None
    assert r["hub_url"] is None and r["peak_vram_gib"] is None  # CPU, no hub
    assert (out / "adapter_config.json").is_file()
    assert (out / "adapter_model.safetensors").is_file()
    assert json.loads((out / "adapter_config.json").read_text())["r"] == 16


def test_hub_repo_without_token_fails_fast(py, tiny, tmp_path):
    model, data = tiny
    spec = JobSpec(base_model=str(model), dataset=str(data), max_steps=1, hub_repo="me/x",
                   output_dir=str(tmp_path)).to_train_json()
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    proc = subprocess.run([py, str(TRAIN_PY)], capture_output=True, text=True,
                          env={**env, **offline_env(), "GPU_AGENT_SPEC": json.dumps(spec)})
    assert proc.returncode == 1
    err = [e for e in map(parse_event, proc.stdout.splitlines()) if e and e["event"] == "error"]
    assert err and "HF_TOKEN" in err[0]["error"]
