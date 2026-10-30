# SPDX-License-Identifier: AGPL-3.0-only
"""Overlapping crop inference and one ByteTrack update in original coordinates.

Tiled inference is established prior work. NMS is not proof of actor identity.
"""

import math

import numpy as np

from roadsense.detection.detector import YoloProposals
from roadsense.detection.tracker import Detection


def tile_windows(
    width: int, height: int, grid: int = 2, overlap: float = 0.2
) -> list[tuple[int, int, int, int]]:
    """Include a full-frame view plus overlapping crops, deduplicated on tiny inputs."""
    if width < 1 or height < 1 or grid not in {1, 2} or not math.isfinite(overlap) or not 0 <= overlap < 0.5:
        raise ValueError("Positive dimensions, grid 1/2 and overlap in [0,0.5) required")
    full = (0, 0, width, height)
    if grid == 1:
        return [full]
    w, h = math.ceil(width / (2 - overlap)), math.ceil(height / (2 - overlap))
    crops = [(x, y, x + w, y + h) for y in [0, height - h] for x in [0, width - w]]
    return list(dict.fromkeys([full, *crops]))


def merge_detections(detections: list[Detection], iou_threshold: float = 0.6) -> list[Detection]:
    """Deterministic class-agnostic IoU NMS after original-coordinate restoration.

    Competing class predictions on one object are suppressed. Real overlapping
    actors can also be suppressed; seam duplicates can survive. Review is needed.
    """
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("NMS threshold must be in (0,1]")
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        if (
            not np.isfinite([*d.bbox, d.confidence]).all()
            or not 0 <= d.confidence <= 1
            or x1 >= x2
            or y1 >= y2
        ):
            raise ValueError("NMS requires finite nondegenerate boxes and valid confidence")
    ordered = sorted(detections, key=lambda d: (-d.confidence, d.bbox, d.class_name))
    kept = []
    for detection in ordered:
        a = detection.bbox
        duplicate = False
        for previous in kept:
            b = previous.bbox
            area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
            if area / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


class TiledYoloProposals:
    """Reuse one local detector across crops; do not retain raw frames."""

    def __init__(
        self,
        weights: str,
        confidence: float = 0.1,
        device: str = "cpu",
        inference_size: int = 640,
        grid: int = 2,
        overlap: float = 0.2,
        nms_iou: float = 0.6,
    ):
        tile_windows(100, 100, grid, overlap)
        merge_detections([], nms_iou)
        self.detector = YoloProposals(weights, confidence, device, inference_size)
        self.grid, self.overlap, self.nms_iou = grid, overlap, nms_iou
        self.last_diagnostics: dict = {}
        self.diagnostics = {
            "processed_frames": 0,
            "model_calls": 0,
            "raw_proposals": 0,
            "merged_proposals": 0,
            "nms_suppressed": 0,
        }

    def detect(self, image: np.ndarray, timestamp: float) -> list[Detection]:
        height, width = image.shape[:2]
        views = tile_windows(width, height, self.grid, self.overlap)
        proposals = []
        for x1, y1, x2, y2 in views:
            crop = np.ascontiguousarray(image[y1:y2, x1:x2])
            for d in self.detector.detect(crop, timestamp):
                a, b, c, e = d.bbox
                box = (
                    max(0.0, a) + x1,
                    max(0.0, b) + y1,
                    min(float(x2 - x1), c) + x1,
                    min(float(y2 - y1), e) + y1,
                )
                proposals.append(Detection(box, d.class_name, d.confidence))
        # Preserve order/NMS at grid one to compare directly with the callback backend.
        merged = merge_detections(proposals, self.nms_iou) if self.grid != 1 else proposals
        self.last_diagnostics = {
            "inference_views": len(views),
            "raw_proposals": len(proposals),
            "merged_proposals": len(merged),
            "nms_suppressed": len(proposals) - len(merged),
        }
        self.diagnostics["processed_frames"] += 1
        self.diagnostics["model_calls"] += len(views)
        for field in ("raw_proposals", "merged_proposals", "nms_suppressed"):
            self.diagnostics[field] += self.last_diagnostics[field]
        return merged


class TiledYoloByteTrack:
    """Merged full-image proposals feed one persistent upstream ByteTrack instance."""

    def __init__(
        self,
        weights: str,
        confidence: float = 0.1,
        device: str = "cpu",
        inference_size: int = 640,
        grid: int = 2,
        overlap: float = 0.2,
        nms_iou: float = 0.6,
    ):
        self.proposals = TiledYoloProposals(
            weights, confidence, device, inference_size, grid, overlap, nms_iou
        )
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        self.boxes_type = Boxes
        self.tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))
        self.class_ids = {name: int(i) for i, name in self.proposals.detector.model.names.items()}

    def detect(self, image: np.ndarray, timestamp: float) -> list[Detection]:
        detections = self.proposals.detect(image, timestamp)
        data = np.array(
            [[*d.bbox, d.confidence, self.class_ids[d.class_name]] for d in detections], dtype=np.float32
        ).reshape(-1, 6)
        tracks = self.tracker.update(self.boxes_type(data, image.shape[:2]), image)
        names = self.proposals.detector.model.names
        return [Detection(tuple(map(float, t[:4])), names[int(t[6])], float(t[5]), int(t[4])) for t in tracks]

    def reset(self) -> None:
        self.tracker.reset()

    @property
    def diagnostics(self) -> dict:
        return dict(self.proposals.diagnostics)
