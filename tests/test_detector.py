# SPDX-License-Identifier: AGPL-3.0-only
"""Resolution forwarding must preserve the detector's original-image coordinates."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from roadsense.config import PipelineConfig
from roadsense.detection.detector import YoloByteTrack, YoloProposals


def test_resolution_validation_and_legacy_default():
    assert PipelineConfig.model_validate({}).inference_size == 640
    assert PipelineConfig(inference_size=960).inference_size == 960
    for invalid in [0, 319, 641, 1952]:
        with pytest.raises(ValidationError):
            PipelineConfig(inference_size=invalid)


def test_yolo_resolution_forwarding_without_coordinate_rescaling(tmp_path, monkeypatch):
    class Tensor:
        def __init__(self, values):
            self.values = np.array(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Model:
        names = {2: "car"}

        def __init__(self, _):
            self.kwargs = None

        def track(self, image, **kwargs):
            self.kwargs = kwargs
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=Tensor([[100, 200, 300, 400]]),
                        cls=Tensor([2]),
                        conf=Tensor([0.8]),
                        id=Tensor([7]),
                    )
                )
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model, settings={}))
    weights = tmp_path / "weights.pt"
    weights.touch()
    detector = YoloByteTrack(str(weights), inference_size=960)
    result = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8), 1)
    assert detector.model.kwargs["imgsz"] == 960
    assert detector.model.kwargs["tracker"] == "bytetrack.yaml"
    assert result[0].bbox == (100, 200, 300, 400)
    assert result[0].track_id == 7


def test_raw_proposals_use_predict_and_preserve_unassigned_detections(tmp_path, monkeypatch):
    class Tensor:
        def __init__(self, values):
            self.values = np.array(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Model:
        names = {2: "car"}

        def __init__(self, _):
            self.kwargs = None

        def predict(self, image, **kwargs):
            self.kwargs = kwargs
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=Tensor([[10, 20, 30, 40]]),
                        cls=Tensor([2]),
                        conf=Tensor([0.2]),
                        id=None,
                    )
                )
            ]

        def track(self, *args, **kwargs):
            raise AssertionError("Raw proposal model must never register tracking callbacks")

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model, settings={}))
    path = tmp_path / "weights.pt"
    path.touch()
    detector = YoloProposals(str(path), inference_size=960)
    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), 1)
    assert result[0].track_id is None and result[0].confidence == 0.2
    assert detector.model.kwargs["imgsz"] == 960
    assert "persist" not in detector.model.kwargs and "tracker" not in detector.model.kwargs
