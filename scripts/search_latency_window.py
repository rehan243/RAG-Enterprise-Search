# lightweight metrics you can cron and ship to cloudwatch or a tiny sqlite
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List


@dataclass
class LatencyWindow:
    # ring buffer so memory stays flat in long running workers
    maxlen: int = 5000
    samples_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))

    def observe(self, ms: float) -> None:
        if self.samples_ms.maxlen != self.maxlen:
            self.samples_ms = deque(self.samples_ms, maxlen=self.maxlen)
        self.samples_ms.append(float(ms))

    def snapshot(self) -> dict:
        if not self.samples_ms:
            return {"n": 0, "p50": None, "p95": None}
        xs = sorted(self.samples_ms)
        n = len(xs)

        def pct(p: float) -> float:
            if n == 1:
                return xs[0]
            idx = int(math.ceil(p * n)) - 1
            return xs[max(0, min(n - 1, idx))]

        return {"n": n, "p50": pct(0.50), "p95": pct(0.95)}


def moving_bad_rate(flags: Iterable[bool], window: int = 200) -> float:
    buf: List[bool] = list(flags)[-window:]
    if not buf:
        return 0.0
    return sum(1 for f in buf if f) / len(buf)


def demo() -> None:
    w = LatencyWindow()
    for i in range(300):
        w.observe(10 + (i % 7) * 3 + (0 if i % 11 else 80))
    print(w.snapshot())
    print("bad_rate", moving_bad_rate([False, True, False, True, True]))


if __name__ == "__main__":
    demo()
