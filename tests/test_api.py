# SPDX-License-Identifier: AGPL-3.0-only
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from roadsense.cloud.api import create_app
from roadsense.cloud.client import EventClient
from roadsense.config import Settings

KEY = "test-key-with-at-least-24-characters"


@pytest.fixture
def api(tmp_path):
    app = create_app(Settings(api_key=KEY, database_url=f"sqlite:///{tmp_path}/test.db"))
    with TestClient(app) as client:
        yield client


def event():
    return {
        "session_id": str(uuid4()),
        "event_id": str(uuid4()),
        "camera_id": "c1",
        "location_id": "L1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video_time_s": 1,
        "event_type": "predicted_conflict",
        "risk_score": 0.7,
        "model_version": "rules-0.1",
        "track_ids": [1, 2],
        "classes": ["car", "bicycle"],
        "features": {
            "distance": 10,
            "relative_speed": 4,
            "closing_speed": 4,
            "cpa_distance": 1,
            "time_to_cpa_s": 2,
            "quality": 0.9,
            "measurement_uncertainty": 1,
            "units": "px",
        },
        "explanations": ["CPA=1 px"],
    }


def test_auth_validation_size_and_idempotency(api):
    assert api.get("/health").status_code == 200
    assert api.get("/events").status_code == 401
    headers = {"X-API-Key": KEY}
    assert api.post("/events", json={}, headers=headers).status_code == 422
    payload = event()
    assert api.post("/events", json=payload, headers=headers).json()["created"]
    assert not api.post("/events", json=payload, headers=headers).json()["created"]
    payload["risk_score"] = 0.8
    assert api.post("/events", json=payload, headers=headers).status_code == 409
    assert len(api.get("/events", headers=headers).json()) == 1
    payload["event_id"] = str(uuid4())
    payload["features"]["ttc_s"] = 1
    assert api.post("/events", json=payload, headers=headers).status_code == 422
    assert api.post("/events", content=b"x" * 70000, headers=headers).status_code == 413
    assert api.get("/events?limit=10000", headers=headers).status_code == 422


def test_synthetic_filter_stats_and_deletion(api):
    headers = {"X-API-Key": KEY}
    payload = event()
    payload["synthetic"] = True
    api.post("/events", json=payload, headers=headers)
    assert api.get("/events", headers=headers).json() == []
    assert len(api.get("/events?synthetic=true", headers=headers).json()) == 1
    now = datetime.now(timezone.utc)
    exposure = {
        "camera_id": "c1",
        "location_id": "L1",
        "start": (now - timedelta(hours=1)).isoformat(),
        "end": (now + timedelta(hours=1)).isoformat(),
        "road_users": 100,
        "method": "manual_count",
        "synthetic": True,
    }
    assert api.post("/exposures", json=exposure, headers=headers).status_code == 200
    assert api.post("/exposures", json=exposure, headers=headers).status_code == 409
    stats = api.get(
        "/stats",
        params={
            "start": (now - timedelta(days=1)).isoformat(),
            "end": (now + timedelta(days=1)).isoformat(),
            "synthetic": True,
        },
        headers=headers,
    ).json()
    assert stats["exposure_normalized"]["c1"]["events_per_1000_road_users"] == 10
    assert api.delete("/events/" + payload["event_id"], headers=headers).json()["deleted"] == 1


def test_client_retry_and_remote_https():
    from roadsense.cloud.schemas import TrafficEvent

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503 if len(calls) == 1 else 200, json={"created": True})

    client = EventClient("http://127.0.0.1:8000", KEY, transport=httpx.MockTransport(handler))
    try:
        assert client.submit(TrafficEvent(**event()))["created"]
        assert len(calls) == 2 and client.bytes_attempted > 0
        assert calls[0].content == calls[1].content
    finally:
        client.close()
    with pytest.raises(ValueError, match="HTTPS"):
        EventClient("http://remote.example", KEY)


def test_invalid_numbers_timestamp_unknown_fields_and_retention(api):
    headers = {"X-API-Key": KEY}
    payload = event()
    payload["risk_score"] = 2
    assert api.post("/events", json=payload, headers=headers).status_code == 422
    payload = event()
    payload["timestamp"] = "2026-10-24T10:00:00"
    assert api.post("/events", json=payload, headers=headers).status_code == 422
    payload = event()
    payload["timestamp"] = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    assert api.post("/events", json=payload, headers=headers).status_code == 422
    payload = event()
    payload["raw_frame"] = "forbidden"
    assert api.post("/events", json=payload, headers=headers).status_code == 422
    assert api.post("/events", content=b"{not json", headers=headers).status_code == 422
    assert api.post("/events", content=iter([b"x" * 40000, b"y" * 40000]), headers=headers).status_code == 413


def test_statistics_exact_range_and_uuid_validation(api):
    headers = {"X-API-Key": KEY}
    now = datetime.now(timezone.utc)
    response = api.get(
        "/stats",
        params={"start": now.isoformat(), "end": (now + timedelta(days=366, hours=1)).isoformat()},
        headers=headers,
    )
    assert response.status_code == 422
    assert api.delete("/events/not-a-uuid", headers=headers).status_code == 422


def test_exposure_insertion_serializes_overlap_check(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from roadsense.cloud.database import Base, connect
    from roadsense.cloud.repository import insert_exposure
    from roadsense.cloud.schemas import Exposure

    now = datetime.now(timezone.utc)
    engine, factory = connect(f"sqlite:///{tmp_path}/concurrent.db")
    Base.metadata.create_all(engine)

    def insert_one(_):
        with factory() as session:
            try:
                insert_exposure(
                    session,
                    Exposure(
                        camera_id="c",
                        location_id="L",
                        start=now,
                        end=now + timedelta(hours=1),
                        road_users=10,
                        method="manual_count",
                    ),
                )
                return "created"
            except ValueError:
                return "overlap"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(insert_one, range(2)))
        assert sorted(results) == ["created", "overlap"]
    finally:
        engine.dispose()


def test_statistics_uses_utc_dates_and_matches_exposure_location():
    from roadsense.analytics.aggregation import aggregate

    payload = event()
    payload["timestamp"] = "2026-10-24T00:30:00+02:00"
    exposure = {
        "camera_id": "c1",
        "location_id": "different-location",
        "start": "2026-10-23T00:00:00+00:00",
        "end": "2026-10-25T00:00:00+00:00",
        "road_users": 100,
    }
    result = aggregate([payload], [exposure])
    assert result["daily_counts"] == {"2026-10-23": 1}
    assert result["exposure_normalized"]["c1"]["covered_events"] == 0


def test_configuration_repr_and_client_urls_protect_credentials():
    settings = Settings(api_key=KEY, database_url="postgresql+psycopg://user:private-password@localhost/db")
    assert KEY not in repr(settings) and "private-password" not in repr(settings)
    with pytest.raises(ValueError, match="credentials out"):
        EventClient("https://user:private-password@localhost", KEY)
