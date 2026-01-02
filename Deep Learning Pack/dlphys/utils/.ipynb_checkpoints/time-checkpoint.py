# dlphys/utils/time.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


def now_str(fmt: str = "%Y%m%d-%H%M%S", *, use_utc: bool = False) -> str:
    """
    Return a timestamp string.
    Default: "YYYYMMDD-HHMMSS"

    Args:
        fmt: datetime format string
        use_utc: if True, use UTC time; else use local time
    """
    if use_utc:
        dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now()
    return dt.strftime(fmt)


def human_time(seconds: float) -> str:
    """
    Convert seconds to a human readable string.
    Examples: "12.3s", "3m 12s", "2h 01m 05s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m:d}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s"


@dataclass
class Timer:
    """
    Simple timer utility, usable as:
      t = Timer().start(); ...; dt = t.elapsed()
    or context manager:
      with Timer() as t: ...
      print(t.elapsed())
    """
    _t0: Optional[float] = None

    def start(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def elapsed(self) -> float:
        if self._t0 is None:
            raise RuntimeError("Timer not started. Call start() first.")
        return time.perf_counter() - self._t0

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        # do nothing; user can read elapsed()
        return None
