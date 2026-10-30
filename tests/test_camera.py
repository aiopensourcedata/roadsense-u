# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from roadsense.camera.sources import parse_source
from roadsense.camera.video_reader import VideoReader
from roadsense.config import PipelineConfig


def test_unavailable_camera(monkeypatch):
    class Unavailable:
        def open(self, *args):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr("roadsense.camera.video_reader.cv2.VideoCapture", Unavailable)
    with pytest.raises(RuntimeError, match="Cannot open"):
        VideoReader(999)


def test_invalid_source_and_synthetic_guard():
    with pytest.raises(ValueError):
        parse_source("not-a-video.mp4")
    with pytest.raises(ValueError):
        PipelineConfig(detector="synthetic")
    assert parse_source("0") == 0


def test_empty_and_corrupt_video_rejected(monkeypatch):
    import cv2

    class EmptyCapture:
        def open(self, *args):
            pass

        def isOpened(self):
            return True

        def release(self):
            pass

        def read(self):
            return False, None

        def get(self, prop):
            return 20 if prop == cv2.CAP_PROP_FPS else 100

    monkeypatch.setattr("roadsense.camera.video_reader.cv2.VideoCapture", EmptyCapture)
    with pytest.raises(RuntimeError, match="no decodable frames"):
        list(VideoReader("empty.avi"))


def test_live_reconnection_is_bounded(monkeypatch):
    import cv2

    class Interrupted:
        def open(self, *args):
            pass

        def isOpened(self):
            return True

        def release(self):
            pass

        def read(self):
            return False, None

        def get(self, prop):
            return 20 if prop == cv2.CAP_PROP_FPS else 0

    monkeypatch.setattr("roadsense.camera.video_reader.cv2.VideoCapture", Interrupted)
    monkeypatch.setattr("roadsense.camera.video_reader.time.sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="exhausted"):
        list(VideoReader("http://127.0.0.1/interrupted", retries=1))


def test_subset_decodes_sequentially_and_preserves_variable_timestamps(monkeypatch):
    import cv2
    import numpy as np

    timestamps = [0.0, 0.03, 0.9, 0.93, 1.0]

    class Capture:
        index = -1
        released = False

        def open(self, *args):
            pass

        def isOpened(self):
            return True

        def release(self):
            self.released = True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return len(timestamps)
            return timestamps[self.index] * 1000

        def read(self):
            self.index += 1
            return (True, np.zeros((4, 4, 3), dtype=np.uint8)) if self.index < 5 else (False, None)

    monkeypatch.setattr("roadsense.camera.video_reader.cv2.VideoCapture", Capture)
    reader = VideoReader("variable.mp4", start_frame=2)
    frames = list(reader)
    assert [f.index for f in frames] == [2, 3, 4]
    assert [f.timestamp for f in frames] == pytest.approx([0.9, 0.93, 1.0])
    assert {f.timestamp_source for f in frames} == {"media_pts"}
    assert reader.cap.released
    with pytest.raises(ValueError, match="beyond"):
        list(VideoReader("variable.mp4", start_frame=5))


def test_invalid_or_live_start_frame_is_rejected_before_open():
    for value in [-1, 1.5, True]:
        with pytest.raises(ValueError, match="nonnegative integer"):
            VideoReader("unused.mp4", start_frame=value)
    with pytest.raises(ValueError, match="prerecorded"):
        VideoReader(999, start_frame=1)


def test_clock_reset_and_repeated_pts_censor_coverage(monkeypatch):
    import cv2
    import numpy as np

    from roadsense.evaluation.coverage import ObservationCoverage

    pts = [0.0, 0.033, 0.067, 0.0, 0.033, 0.033, 0.067]

    class Capture:
        index = -1

        def open(self, *args):
            pass

        def isOpened(self):
            return True

        def release(self):
            pass

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return len(pts)
            return pts[self.index] * 1000

        def read(self):
            self.index += 1
            return (True, np.zeros((4, 4, 3), np.uint8)) if self.index < len(pts) else (False, None)

    monkeypatch.setattr("roadsense.camera.video_reader.cv2.VideoCapture", Capture)
    frames = list(VideoReader("reset.mp4"))
    assert all(b.timestamp > a.timestamp for a, b in zip(frames, frames[1:]))
    assert frames[3].discontinuity and frames[5].discontinuity
    assert frames[3].raw_timestamp_s == 0
    assert frames[4].timestamp_source == "media_pts_new_epoch"
    coverage = ObservationCoverage()
    for frame in frames:
        coverage.observe(frame.timestamp, frame.discontinuity)
    assert not coverage.assess(0.033, 0.1)["future_observation_complete"]
