# SPDX-License-Identifier: AGPL-3.0-only
import json
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from roadsense.cloud.api import create_app
from roadsense.cloud.schemas import TrafficEvent
from roadsense.config import PipelineConfig, Settings
from roadsense.pipeline import run


def test_video_to_computed_events_to_database(tmp_path):
    subprocess.run(
        [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(tmp_path / "data")], check=True
    )
    output = tmp_path / "run"
    report = run(
        str(tmp_path / "data/crossing.avi"), str(output), PipelineConfig(detector="synthetic", synthetic=True)
    )
    assert report["frames"] == 100 and report["events"] >= 1
    rows = [
        TrafficEvent.model_validate_json(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    assert rows[0].features.cpa_distance < 5
    assert rows[0].features.units == "px" and rows[0].features.ttc_s is None
    assert not (output / "annotated.mp4").exists()
    key = "integration-test-secret-at-least-24"
    url = f"sqlite:///{tmp_path}/events.db"
    with TestClient(create_app(Settings(api_key=key, database_url=url))) as api:
        for row in rows:
            assert (
                api.post("/events", json=row.model_dump(mode="json"), headers={"X-API-Key": key}).status_code
                == 200
            )
    # New service instance must see persisted rows.
    with TestClient(create_app(Settings(api_key=key, database_url=url))) as api:
        persisted = api.get("/events?synthetic=true", headers={"X-API-Key": key}).json()
        assert len(persisted) == len(rows)
        assert persisted[0]["event_id"] == str(rows[0].event_id)
    assert json.loads((output / "report.json").read_text())["synthetic"]


def test_ground_plane_pipeline_exports_metric_units(tmp_path):
    import pandas as pd

    subprocess.run(
        [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(tmp_path / "data")], check=True
    )
    output = tmp_path / "calibrated"
    cfg = PipelineConfig(
        detector="synthetic", synthetic=True, calibration=str(tmp_path / "data/calibration.json")
    )
    report = run(str(tmp_path / "data/crossing.avi"), str(output), cfg)
    trajectories = pd.read_csv(output / "trajectories.csv")
    assert set(trajectories.units) == {"m"}
    assert trajectories[trajectories["class"] == "car"].vx.iloc[-1] == pytest.approx(8, abs=0.3)
    assert report["calibration_validation_rmse_m"] < 1e-5


def test_preview_receives_annotated_frames_without_recording(tmp_path):
    class PreviewSink:
        def __init__(self):
            self.frames = []

        def update(self, image):
            self.frames.append(image.copy())

    import cv2
    import numpy as np

    source = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 20, (320, 240))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.full((240, 320, 3), 35, np.uint8))
    writer.release()
    preview = PreviewSink()
    output = tmp_path / "preview"
    report = run(
        str(source), str(output), PipelineConfig(detector="synthetic", synthetic=True), preview=preview
    )
    assert len(preview.frames) == 4
    # The research/synthetic annotation is rendered even when footage recording is disabled.
    assert np.any(preview.frames[0][:40] > 200)
    assert report["live_preview_enabled"]
    assert not (output / "annotated.mp4").exists()


def test_reconnect_resets_alert_cooldown_and_scopes_reused_ids(tmp_path, monkeypatch):
    import numpy as np

    from roadsense.camera.video_reader import Frame
    from roadsense.detection.tracker import Detection

    class Capture:
        def release(self):
            pass

    class Reader:
        fps = 20
        live = False

        def __init__(self, source):
            self.cap = Capture()

        def __iter__(self):
            for i in range(20):
                yield Frame(i, i / 20, np.zeros((100, 100, 3), np.uint8), "test", i == 10)

    class Detector:
        resets = 0

        def reset(self):
            self.resets += 1

        def detect(self, image, timestamp):
            p = -10 + 10 * timestamp
            return [
                Detection((p - 1, -2, p + 1, 0), "car", 1.0, 1),
                Detection((-1, p - 2, 1, p), "bicycle", 1.0, 2),
            ]

    source = tmp_path / "source.avi"
    source.touch()
    monkeypatch.setattr("roadsense.pipeline.VideoReader", Reader)
    detector = Detector()
    output = tmp_path / "reconnect"
    report = run(str(source), str(output), PipelineConfig(), detector_override=detector)
    events = [
        TrafficEvent.model_validate_json(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    assert {e.tracking_epoch for e in events} == {0, 1}
    assert detector.resets == 1 and report["tracking_epochs"] == 2
    assert report["track_segments_not_validated_traffic_volume"] == 4
