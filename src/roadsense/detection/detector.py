# SPDX-License-Identifier: AGPL-3.0-only
"""Pretrained road-user detector and explicitly synthetic fixture detector."""

import os
from pathlib import Path

import cv2
import numpy as np

from roadsense.detection.tracker import Detection, SyntheticTracker

COCO_ROAD_USERS = [0, 1, 2, 3, 5, 7]


def _load_yolo(weights: str):
    """Load local weights with telemetry and automatic dependency installation disabled."""
    if not Path(weights).is_file():
        raise FileNotFoundError(f"Missing weights: {weights}. Download explicitly; see README.")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("outputs/yolo-settings").resolve()))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        from ultralytics import YOLO, settings
    except ImportError as exc:
        raise RuntimeError("Install the vision extra to use YOLO/ByteTrack") from exc
    settings.update({"sync": False})
    return YOLO(weights)


class YoloByteTrack:
    def __init__(self, weights: str, confidence: float = 0.1, device: str = "cpu", inference_size: int = 640):
        if not 320 <= inference_size <= 1920 or inference_size % 32:
            raise ValueError("Inference size must be a multiple of 32 between 320 and 1920")
        self.model, self.confidence, self.device = _load_yolo(weights), confidence, device
        self.inference_size = inference_size

    def detect(self, image: np.ndarray, timestamp: float) -> list[Detection]:
        result = self.model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            classes=COCO_ROAD_USERS,
            conf=self.confidence,
            device=self.device,
            imgsz=self.inference_size,
            verbose=False,
        )[0]
        if result.boxes.id is None:
            return []
        return [
            Detection(tuple(map(float, box)), self.model.names[int(cls)], float(conf), int(tid))
            for box, cls, conf, tid in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.id.cpu().numpy(),
            )
        ]

    def reset(self):
        if self.model.predictor is not None:
            for tracker in getattr(self.model.predictor, "trackers", []):
                tracker.reset()


class YoloProposals:
    """Independent detector-only diagnostic; never shares tracker callbacks or state."""

    def __init__(self, weights: str, confidence: float = 0.1, device: str = "cpu", inference_size: int = 640):
        if not 320 <= inference_size <= 1920 or inference_size % 32:
            raise ValueError("Inference size must be a multiple of 32 between 320 and 1920")
        self.model, self.confidence, self.device = _load_yolo(weights), confidence, device
        self.inference_size = inference_size

    def detect(self, image: np.ndarray, timestamp: float) -> list[Detection]:
        result = self.model.predict(
            image,
            classes=COCO_ROAD_USERS,
            conf=self.confidence,
            device=self.device,
            imgsz=self.inference_size,
            verbose=False,
        )[0]
        return [
            Detection(tuple(map(float, box)), self.model.names[int(cls)], float(conf))
            for box, cls, conf in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            )
        ]


class SyntheticDetector:
    """Color components on generated fixtures only; these are not pretrained detections."""

    def __init__(self):
        self.tracker = SyntheticTracker()

    def detect(self, image: np.ndarray, timestamp: float) -> list[Detection]:
        detections = []
        for channel, name in [(2, "car"), (1, "bicycle")]:
            other = [c for c in range(3) if c != channel]
            mask = ((image[:, :, channel] > 160) & (image[:, :, other].max(axis=2) < 100)).astype("uint8")
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for x, y, w, h, area in stats[1:count]:
                if area >= 80:
                    detections.append(Detection((float(x), float(y), float(x + w), float(y + h)), name, 1.0))
        return self.tracker.update(detections, timestamp)

    def reset(self):
        self.tracker = SyntheticTracker()
