# SPDX-License-Identifier: AGPL-3.0-only
"""Crop geometry/merge controls are synthetic, not traffic accuracy validation."""

import numpy as np
import pytest
from pydantic import ValidationError

from roadsense.config import PipelineConfig
from roadsense.detection.tiled import TiledYoloProposals, merge_detections, tile_windows
from roadsense.detection.tracker import Detection


def test_windows_cover_original_without_out_of_bounds_or_resize():
    windows = tile_windows(101, 77)
    assert windows[0] == (0, 0, 101, 77) and len(windows) == 5
    covered = np.zeros((77, 101), dtype=bool)
    for x1, y1, x2, y2 in windows[1:]:
        assert 0 <= x1 < x2 <= 101 and 0 <= y1 < y2 <= 77
        covered[y1:y2, x1:x2] = True
    assert covered.all()
    assert tile_windows(101, 77, grid=1) == [(0, 0, 101, 77)]
    assert tile_windows(1, 1) == [(0, 0, 1, 1)]
    for overlap in [-0.1, 0.5, float("nan")]:
        with pytest.raises(ValueError):
            tile_windows(100, 100, overlap=overlap)


def test_nms_suppresses_competing_classes_but_retains_distinct_partial_overlap():
    a = Detection((10, 10, 30, 30), "car", 0.9)
    duplicate = Detection((10, 10, 30, 30), "truck", 0.8)
    other = Detection((23, 10, 43, 30), "car", 0.7)
    assert merge_detections([other, duplicate, a]) == [a, other]
    assert merge_detections([a, duplicate, other]) == [a, other]
    with pytest.raises(ValueError):
        merge_detections([Detection((0, 0, 0, 1), "car", 0.9)])


def test_offsets_restore_one_actor_across_four_views_and_deduplicate(monkeypatch):
    class Fake:
        def __init__(self, *args):
            pass

        def detect(self, image, timestamp):
            x, y = image[0, 0, :2]
            return [Detection((45 - int(x), 35 - int(y), 55 - int(x), 45 - int(y)), "car", 0.9)]

    monkeypatch.setattr("roadsense.detection.tiled.YoloProposals", Fake)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(100)
    image[:, :, 1] = np.arange(80)[:, None]
    detector = TiledYoloProposals("synthetic-test-only")
    result = detector.detect(image, 1)
    assert len(result) == 1 and result[0].bbox == (45, 35, 55, 45)
    assert result[0].track_id is None
    assert detector.diagnostics["model_calls"] == 5
    assert detector.diagnostics["nms_suppressed"] == 4


def test_tiling_is_explicit_and_legacy_configs_remain_standard():
    assert not PipelineConfig.model_validate({}).tiling.enabled
    assert PipelineConfig(tiling={"enabled": True}).tiling.grid == 2
    with pytest.raises(ValidationError):
        PipelineConfig(tiling={"overlap": 1})
    with pytest.raises(ValidationError):
        PipelineConfig(detector="synthetic", synthetic=True, tiling={"enabled": True})
