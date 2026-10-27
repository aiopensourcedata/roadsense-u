# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded live reconnection and monotonic media timestamps."""

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray
    timestamp_source: str
    discontinuity: bool = False


class VideoReader:
    def __init__(self, source: int | str, retries: int = 3, start_frame: int = 0):
        if retries < 0:
            raise ValueError("Reconnect retries cannot be negative")
        self.source, self.retries = source, retries
        self.live = isinstance(source, int) or str(source).startswith(("http://", "https://", "rtsp://"))
        if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
            raise ValueError("Start frame must be a nonnegative integer")
        if self.live and start_frame:
            raise ValueError("Start frame is available only for prerecorded files")
        self.start_frame = start_frame
        self.cap = self._open()
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.live and (not np.isfinite(self.fps) or self.fps <= 0):
            self.cap.release()
            raise ValueError("File has no valid frame rate; timing measurements are unavailable")
        if self.live and (not np.isfinite(self.fps) or self.fps <= 0):
            self.fps = 30.0  # output rendering only; live motion uses monotonic arrival time

    def _open(self):
        cap = cv2.VideoCapture()
        if isinstance(self.source, str) and self.live:
            cap.open(
                self.source,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
            )
        else:
            cap.open(self.source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Cannot open camera/video source; check device, codec and connectivity")
        return cap

    def __iter__(self):
        start, previous, failures, index = time.monotonic(), -1.0, 0, 0
        discontinuity = False
        try:
            while True:
                ok, image = self.cap.read()
                if not ok or image is None or image.size == 0:
                    if not self.live:
                        if index == 0:
                            raise RuntimeError("Video contains no decodable frames")
                        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        if np.isfinite(count) and count > index:
                            raise RuntimeError(f"Video decoding interrupted at frame {index} of {count:.0f}")
                        if index <= self.start_frame:
                            raise ValueError("Start frame is beyond the decoded video")
                        break
                    failures += 1
                    if failures > self.retries:
                        raise RuntimeError("Live source interrupted; bounded reconnect attempts exhausted")
                    log.warning("camera_reconnect", extra={"attempt": failures})
                    self.cap.release()
                    time.sleep(0.2 * failures)
                    try:
                        self.cap = self._open()
                    except RuntimeError:
                        continue
                    discontinuity = True
                    continue
                failures = 0
                if self.live:
                    timestamp, origin = time.monotonic() - start, "monotonic_arrival"
                else:
                    pts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    timestamp, origin = (
                        (pts, "media_pts")
                        if np.isfinite(pts) and pts > previous
                        else (index / self.fps, "frame_rate_fallback")
                    )
                    if timestamp <= previous:
                        timestamp, origin = previous + 1 / self.fps, "frame_rate_fallback"
                # Decode sequentially rather than seek: variable-rate sources can
                # otherwise return a different frame or timestamp than requested.
                if index >= self.start_frame:
                    yield Frame(index, timestamp, image, origin, discontinuity)
                previous, index, discontinuity = timestamp, index + 1, False
        finally:
            self.cap.release()
