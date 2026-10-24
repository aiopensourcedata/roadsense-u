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
