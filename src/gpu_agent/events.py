"""Parse the machine-readable event lines that train/train.py prints.

Standard library only: this module also runs inside provider containers.
"""

from __future__ import annotations

import json
from typing import Any

EVENT_PREFIX = "GPU_AGENT_EVENT "  # must match train/train.py


def parse_event(line: str) -> dict[str, Any] | None:
    """Return the event dict for an event line, or None for ordinary log output."""
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        obj = json.loads(line[len(EVENT_PREFIX):])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and "event" in obj else None
