# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit small official weight download with size cap, disk check and provenance."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
EXPECTED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
CHECKPOINTS = {
    "yolo11n": EXPECTED_SHA256,
    "yolo11s": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default="yolo11n")
    args = parser.parse_args()
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{args.model}.pt"
    expected_sha256 = CHECKPOINTS[args.model]
    target = Path("data") / f"{args.model}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit("Weights already exist; remove explicitly before redownloading")
    if shutil.disk_usage(target.parent).free < 100 * 1024**2:
        raise SystemExit("Need at least 100 MiB free for this small download")
    # Fixed HTTPS upstream URL; bounded read and pinned checksum below.
    with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310
        if not response.geturl().startswith("https://"):
            raise SystemExit("Refusing a non-HTTPS download redirect")
        body = response.read(30 * 1024**2 + 1)
    if len(body) > 30 * 1024**2 or len(body) < 1000000:
        raise SystemExit("Unexpected model size; no weights saved")
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise SystemExit("Model checksum differs from the reviewed upstream artifact; no weights saved")
    target.write_bytes(body)
    provenance = {
        "url": url,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "license": "Ultralytics AGPL-3.0 / enterprise; upstream terms apply",
        "checksum_note": "Pinned to the initially reviewed official download; not an independent author signature",
    }
    target.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))
