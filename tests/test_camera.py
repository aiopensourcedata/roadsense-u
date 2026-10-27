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
