# SPDX-License-Identifier: AGPL-3.0-only
"""Small gated assignment tracker used only for synthetic fixtures.

Real video uses Ultralytics ByteTrack. Neither tracker guarantees identity through occlusion.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    class_name: str
    confidence: float
    track_id: int | None = None

    @property
    def position(self) -> np.ndarray:
        x1, _, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, y2])


class SyntheticTracker:
    def __init__(self, gate_px: float = 80, max_gap: float = 0.5):
        self.gate_px, self.max_gap = gate_px, max_gap
        self.tracks: dict[int, tuple] = {}
        self.next_id = 1

    def update(self, detections: list[Detection], timestamp: float) -> list[Detection]:
        self.tracks = {k: v for k, v in self.tracks.items() if timestamp - v[1] <= self.max_gap}
        ids = list(self.tracks)
        if ids and detections:
            cost = np.full((len(ids), len(detections)), 1e6)
            for i, tid in enumerate(ids):
                point, t, velocity, cls = self.tracks[tid]
                predicted = point + velocity * (timestamp - t)
                for j, det in enumerate(detections):
                    if cls == det.class_name:
                        cost[i, j] = np.linalg.norm(predicted - det.position)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] <= self.gate_px:
                    detections[j].track_id = ids[i]
        for det in detections:
            velocity = np.zeros(2)
            if det.track_id is None:
                det.track_id, self.next_id = self.next_id, self.next_id + 1
            elif det.track_id in self.tracks:
                p, t, v, _ = self.tracks[det.track_id]
                dt = timestamp - t
                velocity = 0.5 * v + 0.5 * (det.position - p) / max(dt, 1e-6)
            self.tracks[det.track_id] = (det.position, timestamp, velocity, det.class_name)
        return detections
