"""Hand-maintained list prices. Dated so quotes can be flagged stale (full catalog lands in M3)."""

from __future__ import annotations

# Modal list prices, per-second billing. Source: modal.com/pricing, captured in M0 (docs/M0-spikes.md).
MODAL_PRICES_AS_OF = "2026-11-04"
MODAL_USD_PER_HOUR: dict[str, float] = {
    "T4": 0.59,
    "L4": 0.80,
    "A10": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
}

# Names people (and Modal's own device names) use for the same GPU.
_ALIASES = {"A10G": "A10", "A100": "A100-40GB", "A100-40": "A100-40GB", "A100-80": "A100-80GB"}


def normalize_modal_gpu(name: str) -> str:
    """Map a user or device GPU name onto a key of MODAL_USD_PER_HOUR. Raises KeyError if unknown."""
    key = name.strip().upper()
    key = _ALIASES.get(key, key)
    if key not in MODAL_USD_PER_HOUR:
        raise KeyError(f"unknown Modal GPU {name!r}; known: {', '.join(MODAL_USD_PER_HOUR)}")
    return key
