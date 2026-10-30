# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in real dependency boundary; generated arrays require no traffic media."""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ROADSENSE_VISION_INTEGRATION") != "1", reason="Opt-in pinned vision job"
)


def test_real_tracker_boxes_empty_persistence_and_reset(monkeypatch):
    from roadsense.detection import tiled
    from roadsense.detection.tracker import Detection

    class Proposals:
        def __init__(self, *args):
            from types import SimpleNamespace

            self.detector = SimpleNamespace(model=SimpleNamespace(names={2: "car"}))
            self.empty = False

        def detect(self, image, timestamp):
            return [] if self.empty else [Detection((10, 10, 30, 30), "car", 0.9)]

    monkeypatch.setattr(tiled, "TiledYoloProposals", Proposals)
    adapter = tiled.TiledYoloByteTrack("unused")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    first = adapter.detect(image, 0)
    second = adapter.detect(image, 0.1)
    assert len(first) == len(second) == 1 and first[0].track_id == second[0].track_id
    adapter.proposals.empty = True
    assert adapter.detect(image, 0.2) == []
    adapter.reset()
    assert adapter.tracker.frame_id == 0 and not adapter.tracker.tracked_stracks


def test_real_yolo_blank_inference_and_reset():
    from roadsense.detection.detector import YoloByteTrack
    from roadsense.provenance import file_sha256

    weights = Path("data/yolo11n.pt")
    assert weights.is_file(), "Download the pinned permitted checkpoint explicitly before this job"
    assert file_sha256(weights) == "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
    detector = YoloByteTrack(str(weights))
    assert detector.detect(np.zeros((320, 320, 3), dtype=np.uint8), 0) == []
    assert len(detector.model.predictor.trackers) == 1
    detector.reset()
    assert detector.model.predictor.trackers[0].frame_id == 0
