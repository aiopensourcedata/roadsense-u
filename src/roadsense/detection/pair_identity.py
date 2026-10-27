# SPDX-License-Identifier: AGPL-3.0-only
"""Causal evidence of ambiguous object identity; never merge or delete tracks.

Sustained overlap and nearly constant relative box position can indicate duplicate
tracks, but also distinct occluded objects. Abstention is therefore an ambiguity
flag, not a duplicate-object classification or a collision detector.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from roadsense.config import PairIdentityConfig
from roadsense.detection.tracker import Detection


@dataclass
class IdentityAssessment:
    ambiguous: bool = False
    measurements: dict = field(default_factory=dict)


class PairIdentityGuard:
    """Use only current/past boxes, with bounded history and no identity reassignment."""

    def __init__(self, config: PairIdentityConfig | None = None):
        self.config = config or PairIdentityConfig()
        self.history: dict[tuple[int, int, str], deque] = {}

    def clear(self) -> None:
        self.history.clear()

    def begin_frame(self, timestamp: float) -> None:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Identity timestamp must be finite")
        expired = [key for key, h in self.history.items() if timestamp - h[-1][0] > self.config.max_gap_s]
        for key in expired:
            del self.history[key]

    def assess(self, a: Detection, b: Detection, timestamp: float) -> IdentityAssessment:
        cfg = self.config
        if not cfg.enabled or a.class_name != b.class_name:
            return IdentityAssessment()
        if a.track_id is None or b.track_id is None or a.track_id == b.track_id:
            raise ValueError("Pair identity requires two distinct assigned track IDs")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Identity timestamp must be finite")
        if a.track_id > b.track_id:
            a, b = b, a
        boxes = np.asarray([a.bbox, b.bbox], dtype=float)
        sizes = boxes[:, 2:] - boxes[:, :2]
        if not np.isfinite(boxes).all() or (sizes <= 0).any():
            raise ValueError("Identity requires finite positive-area boxes")
        overlap = np.maximum(
            0, np.minimum(boxes[0, 2:], boxes[1, 2:]) - np.maximum(boxes[0, :2], boxes[1, :2])
        )
        intersection = float(np.prod(overlap))
        iou = intersection / (float(np.prod(sizes, axis=1).sum()) - intersection)
        key = (a.track_id, b.track_id, a.class_name)
        if key not in self.history and iou < cfg.min_iou:
            return IdentityAssessment()
        h = self.history.setdefault(key, deque(maxlen=128))
        if h and timestamp <= h[-1][0]:
            raise ValueError("Pair identity timestamps must increase")
        if h and timestamp - h[-1][0] > cfg.max_gap_s:
            h.clear()
        scale = float(np.linalg.norm(sizes, axis=1).mean())
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        # Dimensionless relative position; its slope measures disagreement in box motion/scale.
        offset = (centers[0] - centers[1]) / scale
        h.append((timestamp, iou, offset))
        while h and timestamp - h[0][0] > cfg.window_s:
            h.popleft()
        measurements = {
            "iou": iou,
            "center_separation_box_diagonals": float(np.linalg.norm(offset)),
            "samples": len(h),
        }
        span = timestamp - h[0][0]
        if len(h) < cfg.min_samples or span < cfg.min_span_s:
            return IdentityAssessment(False, measurements)
        times = np.array([v[0] for v in h])
        times -= times.mean()
        offsets = np.array([v[2] for v in h])
        slope = (times[:, None] * offsets).sum(axis=0) / np.dot(times, times)
        drift = float(np.linalg.norm(slope))
        residual = float(
            np.sqrt(np.mean(np.sum((offsets - offsets.mean(axis=0) - times[:, None] * slope) ** 2, axis=1)))
        )
        fraction = sum(v[1] >= cfg.min_iou for v in h) / len(h)
        measurements.update(
            span_s=span,
            overlap_fraction=fraction,
            relative_offset_drift_per_s=drift,
            relative_offset_residual=residual,
        )
        ambiguous = (
            iou >= cfg.min_iou
            and np.linalg.norm(offset) <= cfg.max_center_separation
            and fraction >= cfg.min_overlap_fraction
            and drift <= cfg.max_offset_drift_per_s
            and residual <= cfg.max_offset_residual
        )
        return IdentityAssessment(bool(ambiguous), measurements)
