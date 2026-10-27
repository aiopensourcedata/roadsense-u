# SPDX-License-Identifier: AGPL-3.0-only
"""Retrospective observation coverage, separate from causal risk inference."""

import math
from bisect import bisect_right


class ObservationCoverage:
    """Store contiguous observed intervals, not every frame; gaps censor both labels.

    Timestamp continuity does not prove visibility, valid labels or genuine capture
    time. Callers must separately audit those prerequisites. A positive outcome seen
    before a gap is still excluded here under the conservative full-horizon protocol.
    """

    def __init__(self, max_gap_s: float = 0.5):
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("Coverage max gap must be positive and finite")
        self.max_gap_s = max_gap_s
        self.segments: list[list[float]] = []
        self.starts: list[float] = []
        self.frames = 0

    def observe(self, timestamp: float, discontinuity: bool = False) -> None:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Observation timestamps must be finite and nonnegative")
        if self.segments and timestamp <= self.segments[-1][1]:
            raise ValueError("Observation timestamps must increase")
        if not self.segments or discontinuity or timestamp - self.segments[-1][1] > self.max_gap_s:
            self.segments.append([timestamp, timestamp])
            self.starts.append(timestamp)
        else:
            self.segments[-1][1] = timestamp
        self.frames += 1

    def assess(self, prediction_time_s: float, horizon_s: float = 3) -> dict:
        if not math.isfinite(prediction_time_s) or prediction_time_s < 0:
            raise ValueError("Prediction timestamp must be finite and nonnegative")
        if not math.isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError("Prediction horizon must be positive and finite")
        index = bisect_right(self.starts, prediction_time_s) - 1
        end = prediction_time_s + horizon_s
        if index < 0 or prediction_time_s > self.segments[index][1] + 1e-9:
            reason = "prediction_outside_observation"
        elif end <= self.segments[index][1] + 1e-9:
            reason = "complete_observed_horizon"
        elif index < len(self.segments) - 1:
            reason = "timestamp_gap_in_future_horizon"
        else:
            reason = "insufficient_future_observation"
        return {
            "prediction_time_s": prediction_time_s,
            "horizon_s": horizon_s,
            "future_observation_complete": reason == "complete_observed_horizon",
            "reason": reason,
            "max_gap_s": self.max_gap_s,
            "retrospective": True,
        }
