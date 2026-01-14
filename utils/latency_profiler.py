from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class CudaEventProfiler:
    """
    Lightweight CUDA-event profiler.

    - Records named CUDA events on the current stream.
    - Computes elapsed times (seconds) between named marks.
    - Synchronizes only once (on a chosen end event).
    """

    enabled: bool = True
    _events: Dict[str, torch.cuda.Event] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        if not torch.cuda.is_available():
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._events[name] = ev

    def has(self, name: str) -> bool:
        return name in self._events

    def synchronize(self, name: str) -> None:
        if not self.enabled:
            return
        ev = self._events.get(name)
        if ev is None:
            return
        ev.synchronize()

    def elapsed_s(self, start: str, end: str) -> Optional[float]:
        if not self.enabled or not torch.cuda.is_available():
            return None
        s = self._events.get(start)
        e = self._events.get(end)
        if s is None or e is None:
            return None
        # elapsed_time returns milliseconds
        return float(s.elapsed_time(e)) / 1000.0

