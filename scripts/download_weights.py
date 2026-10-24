# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit small official weight download with size cap, disk check and provenance."""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
EXPECTED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"

if __name__ == "__main__":
    target = Path("data/yolo11n.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit("Weights already exist; remove explicitly before redownloading")
    if shutil.disk_usage(target.parent).free < 100 * 1024**2:
        raise SystemExit("Need at least 100 MiB free for this small download")
    # Fixed HTTPS upstream URL; bounded read and pinned checksum below.
    with urllib.request.urlopen(URL, timeout=30) as response:  # nosec B310
        if not response.geturl().startswith("https://"):
            raise SystemExit("Refusing a non-HTTPS download redirect")
        body = response.read(30 * 1024**2 + 1)
    if len(body) > 30 * 1024**2 or len(body) < 1000000:
        raise SystemExit("Unexpected model size; no weights saved")
    if hashlib.sha256(body).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("Model checksum differs from the reviewed upstream artifact; no weights saved")
    target.write_bytes(body)
    provenance = {
        "url": URL,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "license": "Ultralytics AGPL-3.0 / enterprise; upstream terms apply",
        "checksum_note": "Pinned to the initially reviewed official download; not an independent author signature",
    }
    target.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))
