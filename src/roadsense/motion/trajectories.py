# SPDX-License-Identifier: AGPL-3.0-only
"""Causal trailing least-squares motion estimates with gap and discontinuity flags."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class MotionState:
    track_id: int
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    quality: float
    residual: float
    units: str
    flags: list[str]
    speed_change: float = 0.0
    direction_change_deg: float = 0.0


class Trajectories:
    def __init__(self, window_s: float = 1.0, max_gap_s: float = 0.5, min_samples: int = 4):
        if not np.isfinite([window_s, max_gap_s]).all() or min(window_s, max_gap_s) <= 0 or min_samples < 2:
            raise ValueError("Motion windows must be positive and require at least two samples")
        self.window_s, self.max_gap_s, self.min_samples = window_s, max_gap_s, min_samples
        self.history: dict[int, deque] = {}
        self.previous: dict[int, MotionState] = {}

    def clear(self):
        self.history.clear()
        self.previous.clear()

    def update(
        self, tid: int, timestamp: float, point: np.ndarray, confidence: float, units: str = "px"
    ) -> MotionState:
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or units not in {"px", "m"} or not 0 <= confidence <= 1:
            raise ValueError("Observation requires a 2D point, px/m units and confidence in [0,1]")
        if not np.isfinite([timestamp, *point, confidence]).all():
            raise ValueError("Nonfinite trajectory observation")
        h = self.history.setdefault(tid, deque())
        flags = []
        prev = self.previous.get(tid)
        if prev and timestamp <= prev.timestamp:
            raise ValueError("Track timestamps must increase")
        if prev and (timestamp - prev.timestamp > self.max_gap_s or prev.units != units):
            h.clear()
            prev = None
            flags.append("gap_or_coordinate_change_reset")
        h.append((timestamp, np.array(point, float)))
        while h and timestamp - h[0][0] > self.window_s:
            h.popleft()
        if len(h) < self.min_samples or timestamp - h[0][0] < 0.1:
            flags.append("insufficient_history")
            state = MotionState(tid, timestamp, point, np.zeros(2), np.zeros(2), 0.0, 0.0, units, flags)
        else:
            times = np.array([p[0] - timestamp for p in h])
            points = np.array([p[1] for p in h])
            design = np.column_stack([np.ones(len(h)), times])
            coef = np.linalg.lstsq(design, points, rcond=None)[0]
            residual = float(np.sqrt(np.mean(np.sum((points - design @ coef) ** 2, axis=1))))
            displacement = float(np.linalg.norm(coef[1]) * max(np.ptp(times), 0.1))
            quality = float(confidence / (1 + residual / max(displacement, 1e-3)))
            acceleration, speed_change, turn = np.zeros(2), 0.0, 0.0
            if prev and prev.quality > 0:
                dt = timestamp - prev.timestamp
                acceleration = (coef[1] - prev.velocity) / dt
                speed_change = (np.linalg.norm(coef[1]) - np.linalg.norm(prev.velocity)) / dt
                if np.linalg.norm(coef[1]) * np.linalg.norm(prev.velocity) > 1e-6:
                    cos = np.dot(coef[1], prev.velocity) / (
                        np.linalg.norm(coef[1]) * np.linalg.norm(prev.velocity)
                    )
                    turn = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            if prev and np.linalg.norm(point - prev.position) > max(10 * residual, 5 * displacement, 5):
                flags.append("possible_id_switch_or_jump")
                quality *= 0.1
            state = MotionState(
                tid,
                timestamp,
                coef[0],
                coef[1],
                acceleration,
                quality,
                residual,
                units,
                flags,
                float(speed_change),
                turn,
            )
        self.previous[tid] = state
        # Prevent unbounded storage for departed tracks.
        for old in [k for k, v in self.previous.items() if timestamp - v.timestamp > 2 * self.window_s]:
            self.previous.pop(old, None)
            self.history.pop(old, None)
        return state
