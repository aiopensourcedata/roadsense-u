# SPDX-License-Identifier: AGPL-3.0-only
"""Smoke-test actual loopback HTML, JPEG and MJPEG serving during video processing."""

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
from generate_synthetic_data import generate

from roadsense.camera.preview import LivePreview
from roadsense.config import PipelineConfig
from roadsense.pipeline import run

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="roadsense-preview-") as directory:
        root = Path(directory)
        generate(str(root / "data"))
        with LivePreview(port=0) as preview, ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                run,
                str(root / "data/crossing.avi"),
                str(root / "output"),
                PipelineConfig(detector="synthetic", synthetic=True, max_frames=40),
                preview=preview,
                realtime=True,
            )
            with preview._condition:
                assert preview._condition.wait_for(lambda: preview._jpeg is not None, timeout=10)
            with httpx.Client(trust_env=False, base_url=preview.url, timeout=5) as client:
                response = client.get("/")
                response.raise_for_status()
                assert "RoadSense-U live preview" in response.text
                response = client.get("/frame.jpg")
                response.raise_for_status()
                assert response.headers["cache-control"] == "no-store"
                import numpy as np

                assert cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR) is not None
                with client.stream("GET", "/stream.mjpg") as response:
                    response.raise_for_status()
                    data = next(response.iter_bytes())
                    assert b"--frame" in data and b"image/jpeg" in data
                assert client.get("/frame.jpg", headers={"Host": "untrusted.example"}).status_code == 403
            report = future.result(timeout=10)
            assert report["frames"] == 40 and not (root / "output/annotated.mp4").exists()
    print("Live preview verified: HTML, decodable JPEG, MJPEG frames, host restriction and no recording.")
