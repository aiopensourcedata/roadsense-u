# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable run manifests, containing hashes rather than private source URLs."""

import hashlib
import importlib.metadata
import json
from datetime import timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_manifest(config, source, session_id, origin, geometry, injected=False, *, wall_origin):
    root = Path(__file__).parent
    code = {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*.py"))}
    packages = {}
    for name in (
        "numpy",
        "opencv-python-headless",
        "ultralytics-opencv-headless",
        "torch",
        "lap",
        "scikit-learn",
        "pydantic",
        "sqlalchemy",
        "httpx",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    tracker = None
    if config.detector == "yolo" and not injected:
        from ultralytics.utils import ROOT

        tracker = __import__("yaml").safe_load((ROOT / "cfg/trackers/bytetrack.yaml").read_text())
    content = {
        "schema_version": 1,
        "session_id": str(session_id),
        "camera_id": config.camera_id,
        "location_id": config.location_id,
        "timestamp_origin": origin,
        "wall_time_origin": wall_origin.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_sha256": file_sha256(Path(source))
        if not geometry["live"] and isinstance(source, str) and Path(source).is_file()
        else None,
        "geometry": geometry,
        "config": config.model_dump(),
        "code_sha256": canonical_digest(code),
        "code_files": code,
        "dependencies": packages,
        "tracker_settings": tracker,
        "detector_injected_for_test": injected,
        "weights_sha256": file_sha256(Path(config.weights))
        if config.detector == "yolo" and not injected
        else None,
        "calibration_sha256": file_sha256(Path(config.calibration)) if config.calibration else None,
        "timing": "media PTS; reset epochs invalidate crossing horizons"
        if geometry["live"] is False
        else "monotonic decode-arrival; device capture time and capture age unavailable",
    }
    # Paths are local configuration, never credentials or raw stream URLs.
    content["config"]["weights"] = Path(config.weights).name
    content["config"]["calibration"] = Path(config.calibration).name if config.calibration else None
    return {"sha256": canonical_digest(content), "content": content}
