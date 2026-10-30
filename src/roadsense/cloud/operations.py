# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded process-local HTTP measurements; not distributed SLA statistics."""

import threading
from collections import Counter, deque
from datetime import datetime, timezone

import numpy as np


class Operations:
    def __init__(self):
        self.lock = threading.Lock()
        self.responses = Counter()
        self.latencies = deque(maxlen=10000)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.retention = {"status": "not_run"}

    def observe(self, status: int, elapsed_ms: float) -> None:
        with self.lock:
            self.responses[str(status)] += 1
            self.latencies.append(elapsed_ms)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "started_at": self.started_at,
                "response_counts": dict(self.responses),
                "latency_samples": len(self.latencies),
                "request_p95_ms": float(np.percentile(self.latencies, 95)) if self.latencies else None,
                "retention": dict(self.retention),
                "scope": "This API process only; latest up to 10000 response latencies, not an SLA",
            }
