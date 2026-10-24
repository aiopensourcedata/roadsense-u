# SPDX-License-Identifier: AGPL-3.0-only
"""Official static image repeated as video: detector/codec smoke check, not motion validation."""

import hashlib
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

if __name__ == "__main__":
    destination = Path("data/detector-smoke")
    destination.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    # Fixed official HTTPS example URL; bounded read and decoding below.
    with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310
        if not response.geturl().startswith("https://"):
            raise SystemExit("Refusing a non-HTTPS download redirect")
        body = response.read(2 * 1024**2 + 1)
    if len(body) > 2 * 1024**2:
        raise SystemExit("Unexpected image size")
    image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("Invalid image")
    (destination / "bus.jpg").write_bytes(body)
    h, w = image.shape[:2]
    writer = cv2.VideoWriter(str(destination / "static-bus.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    if not writer.isOpened():
        raise SystemExit("AVI encoder unavailable")
    try:
        for _ in range(20):
            writer.write(image)
    finally:
        writer.release()
    (destination / "provenance.json").write_text(
        json.dumps(
            {
                "url": url,
                "sha256": hashlib.sha256(body).hexdigest(),
                "fixture": "Repeated still image. No real motion or conflict validation.",
                "license": "From Ultralytics AGPL-3.0 repository; upstream terms apply",
            },
            indent=2,
        )
    )
    print("Created static-image detector smoke fixture (not a real traffic video).")
