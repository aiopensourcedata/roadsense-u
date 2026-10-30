# SPDX-License-Identifier: AGPL-3.0-only
"""Interruption, immutable provenance and bounded asynchronous delivery regressions."""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api import KEY, event

from roadsense.camera.video_reader import Frame
from roadsense.cloud.api import create_app
from roadsense.cloud.outbox import DeliveryOutbox
from roadsense.cloud.schemas import TrafficEvent
from roadsense.config import PipelineConfig, Settings
from roadsense.dashboard.artifacts import processed_runs
from roadsense.pipeline import run
from roadsense.provenance import build_manifest


@pytest.mark.parametrize("exception,status", [(KeyboardInterrupt, "interrupted"), (RuntimeError, "failed")])
def test_partial_run_finalized_before_exception(tmp_path, monkeypatch, exception, status):
    class Capture:
        def release(self):
            pass

    class Reader:
        fps, live = 20, False

        def __init__(self, *args, **kwargs):
            self.cap = Capture()

        def __iter__(self):
            yield Frame(0, 0, np.zeros((100, 100, 3), np.uint8), "test")
            raise exception("stop")

    source = tmp_path / "test.avi"
    source.touch()
    monkeypatch.setattr("roadsense.pipeline.VideoReader", Reader)
    output = tmp_path / "run"
    with pytest.raises(exception):
        run(str(source), str(output), PipelineConfig(detector="synthetic", synthetic=True))
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == status and report["frames"] == 1
    assert (output / "event_evaluation.jsonl").exists()
    assert processed_runs(tmp_path) == [output]
    assert not list(output.glob("*.tmp"))


def test_outbox_does_not_block_frame_thread_and_retains_pending(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Client:
        def submit(self, value):
            entered.set()
            assert release.wait(5)

    path = tmp_path / "outbox.sqlite"
    outbox = DeliveryOutbox(path, Client(), capacity=1)
    payload = TrafficEvent.model_validate(event())
    try:
        assert outbox.enqueue(payload)
        assert entered.wait(2)
        assert not outbox.enqueue(TrafficEvent.model_validate(event()))
        start = time.monotonic()
        snapshot = outbox.finish(0.01)
        assert time.monotonic() - start < 0.5
        assert snapshot["pending"] == 1 and snapshot["worker_active"]
        assert snapshot["queue_rejected"] == 1
    finally:
        release.set()
        outbox.thread.join(3)
    assert not outbox.thread.is_alive()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM pending").fetchone()[0] == 0


def test_failed_outbox_restarts_with_same_event_uuid(tmp_path):
    class Failed:
        def submit(self, value):
            raise RuntimeError("offline")

    path = tmp_path / "outbox.sqlite"
    original = TrafficEvent.model_validate(event())
    first = DeliveryOutbox(path, Failed())
    first.enqueue(original)
    first.thread.join(2)
    assert first.finish()["pending"] == 1
    received = []

    class Recovered:
        def submit(self, value):
            received.append(value.event_id)

    second = DeliveryOutbox(path, Recovered())
    assert second.finish(2)["pending"] == 0
    assert received == [original.event_id]


def manifest_event(tmp_path):
    value = event()
    cfg = PipelineConfig(
        detector="synthetic", synthetic=True, camera_id=value["camera_id"], location_id=value["location_id"]
    )
    value["timestamp_origin"] = "ingestion_start_not_capture_time"
    value["run_manifest"] = build_manifest(
        cfg,
        0,
        value["session_id"],
        value["timestamp_origin"],
        {
            "width": 100,
            "height": 100,
            "live": True,
            "image_transform": "decoded_native",
            "camera_pose_version": "unspecified",
        },
        wall_origin=datetime.fromisoformat(value["timestamp"]) - timedelta(seconds=value["video_time_s"]),
    )
    return value


def test_manifest_api_binding_persistence_and_tampering(tmp_path):
    cfg = Settings(api_key=KEY, database_url=f"sqlite:///{tmp_path}/api.db")
    headers = {"X-API-Key": KEY}
    value = manifest_event(tmp_path)
    with TestClient(create_app(cfg)) as api:
        assert api.post("/events", json=value, headers=headers).status_code == 200
        assert api.get("/runs/" + value["session_id"]).status_code == 401
    with TestClient(create_app(cfg)) as api:
        response = api.get("/runs/" + value["session_id"], headers=headers)
        assert response.json() == value["run_manifest"]
        value["run_manifest"]["content"]["geometry"]["width"] = 200
        assert api.post("/events", json=value, headers=headers).status_code == 422
        value = manifest_event(tmp_path)
        value["camera_id"] = "wrong-camera"
        assert api.post("/events", json=value, headers=headers).status_code == 422


def test_calibration_requires_exact_acquisition_binding():
    from roadsense.calibration.ground_plane import GroundPlane

    binding = dict(
        camera_id="c1", width=100, height=100, image_transform="decoded_native", camera_pose_version="v1"
    )
    plane = GroundPlane(
        [[0, 0], [100, 0], [100, 100], [0, 100]],
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        0.1,
        [[25, 25], [75, 75]],
        [[2.5, 2.5], [7.5, 7.5]],
        binding=binding,
    )
    plane.validate_binding("c1", 100, 100, "v1")
    for args in [
        ("c2", 100, 100, "v1"),
        ("c1", 200, 100, "v1"),
        ("c1", 100, 100, "v2"),
        ("c1", 100, 100, "unspecified"),
    ]:
        with pytest.raises(ValueError, match="Calibration requires"):
            plane.validate_binding(*args)
    plane.binding = None
    with pytest.raises(ValueError):
        plane.validate_binding("c1", 100, 100, "v1")


def test_pipeline_outbox_delivers_bound_events_to_persistent_api(tmp_path):
    import subprocess
    import sys

    import httpx

    from roadsense.cloud.client import EventClient

    subprocess.run(
        [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(tmp_path / "input")],
        check=True,
        capture_output=True,
    )
    settings = Settings(api_key=KEY, database_url=f"sqlite:///{tmp_path}/events.db")
    with TestClient(create_app(settings)) as api:

        def handler(request):
            response = api.post(
                request.url.path,
                content=request.content,
                headers={"X-API-Key": KEY, "Content-Type": "application/json"},
            )
            return httpx.Response(response.status_code, json=response.json())

        client = EventClient("http://127.0.0.1:8000", KEY, transport=httpx.MockTransport(handler))
        try:
            report = run(
                str(tmp_path / "input/crossing.avi"),
                str(tmp_path / "run"),
                PipelineConfig(detector="synthetic", synthetic=True, delivery_shutdown_s=3),
                client=client,
            )
        finally:
            client.close()
        assert report["delivery"]["pending"] == 0
        assert report["delivery"]["sent"] == report["events"] >= 1
    with TestClient(create_app(settings)) as api:
        rows = api.get("/events?synthetic=true", headers={"X-API-Key": KEY}).json()
        assert len(rows) == report["events"]
        assert rows[0]["run_manifest"]["sha256"] == report["run_manifest_sha256"]
        assert rows[0]["timestamp_origin"] == "ingestion_start_not_capture_time"
        assert rows[0]["capture_to_alert_ms"] is None


def test_initialization_failure_is_discoverable(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        run(str(tmp_path / "missing.mp4"), str(output), PipelineConfig())
    report = json.loads((output / "report.json").read_text())
    assert report["failure_stage"] == "initialization" and report["status"] == "failed"
    assert processed_runs(tmp_path) == [output]


def test_server_ingestion_time_and_legacy_replay(tmp_path):
    from roadsense.cloud.database import EventRow

    settings = Settings(api_key=KEY, database_url=f"sqlite:///{tmp_path}/legacy.db")
    app = create_app(settings)
    value = event()
    with TestClient(app) as api:
        headers = {"X-API-Key": KEY}
        assert api.post("/events", json=value, headers=headers).status_code == 200
        stored = api.get("/events", headers=headers).json()[0]
        assert stored["ingested_at"] is not None
        with app.state.session_factory() as session:
            row = session.get(EventRow, value["event_id"])
            row.payload = value  # Simulate an original-version record without new default fields.
            session.commit()
        assert not api.post("/events", json=value, headers=headers).json()["created"]
