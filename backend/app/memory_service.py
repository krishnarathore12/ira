"""Lazy singleton for HiMem Memory (heavy startup)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from himem.memory.main import Memory

_BACKEND = Path(__file__).resolve().parent.parent
_HIMEM = _BACKEND / "HiMem"
if str(_HIMEM) not in sys.path:
    sys.path.insert(0, str(_HIMEM))

_memory: Memory | None = None


def get_memory() -> Memory:
    global _memory
    if _memory is None:
        from himem.memory.main import Memory

        from app.config import load_himem_config_dict

        _memory = Memory.from_config(load_himem_config_dict())
    return _memory
