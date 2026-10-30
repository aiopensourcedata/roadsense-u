# SPDX-License-Identifier: AGPL-3.0-only
"""Scope denial, migration/recovery and autonomous queue-recovery boundaries."""

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from test_api import event
from test_lifecycle import manifest_event

from roadsense.cloud.api import create_app
from roadsense.cloud.backup import backup, restore
from roadsense.cloud.database import EventRow, connect
from roadsense.cloud.migrations import migrate, require_current
from roadsense.cloud.outbox import DeliveryOutbox
from roadsense.cloud.schemas import TrafficEvent
from roadsense.config import Settings


def scoped_settings(tmp_path):
    tokens = {
        "admin": "admin-token-for-tests-long-enough",
        "read": "reader-token-for-tests-long-enough",
        "ingest": "ingest-token-for-tests-long-enough",
        "other": "other-camera-admin-test-token-long",
    }
    records = [
        {
            "credential_id": name,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "role": name if name != "other" else "admin",
            "cameras": ["*"] if name == "admin" else ["c1"],
        }
        for name, token in tokens.items()
    ]
    return Settings(
        auth_mode="scoped", service_credentials=records, database_url=f"sqlite:///{tmp_path}/scoped.db"
    ), tokens


def test_camera_and_role_scopes_cover_all_data_routes(tmp_path):
    cfg, tokens = scoped_settings(tmp_path)

    def h(name):
        return {"X-API-Key": tokens[name]}

    with TestClient(create_app(cfg)) as api:
        a, b = event(), event()
        b["camera_id"] = "c2"
        assert api.post("/events", json=a, headers=h("ingest")).status_code == 200
        assert api.post("/events", json=b, headers=h("ingest")).status_code == 403
        assert api.post("/events", json=b, headers=h("admin")).status_code == 200
        assert api.post("/events", json=event(), headers=h("read")).status_code == 403
        assert api.get("/events", headers=h("ingest")).status_code == 403
        assert {row["camera_id"] for row in api.get("/events", headers=h("read")).json()} == {"c1"}
        assert api.get("/events?camera_id=c2", headers=h("read")).status_code == 403
        query = {
            "start": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "end": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        }
        stats = api.get("/stats", params=query, headers=h("read")).json()
        assert stats["event_count"] == 1 and set(stats["by_camera"]) == {"c1"}
        assert api.delete("/events/" + a["event_id"], headers=h("read")).status_code == 403
        assert api.delete("/events/" + b["event_id"], headers=h("other")).status_code == 404
        assert api.post("/maintenance/purge", headers=h("read")).status_code == 403
        assert api.post("/maintenance/purge", headers=h("other")).status_code == 403
        assert api.get("/operations", headers=h("read")).status_code == 403
        assert api.get("/audit", headers=h("read")).status_code == 403
        assert api.get("/operations", headers=h("admin")).json()["auth_mode"] == "scoped"
        logs = api.get("/audit", headers=h("admin")).json()
        assert {r["principal_id"] for r in logs} >= {"admin", "ingest"}
        snapshot = manifest_event(tmp_path)
        snapshot["camera_id"] = "c2"
        snapshot["run_manifest"]["content"]["camera_id"] = "c2"
        snapshot["run_manifest"]["content"]["config"]["camera_id"] = "c2"
        from roadsense.provenance import canonical_digest

        snapshot["run_manifest"]["sha256"] = canonical_digest(snapshot["run_manifest"]["content"])
        assert api.post("/events", json=snapshot, headers=h("admin")).status_code == 200
        assert api.get("/runs/" + snapshot["session_id"], headers=h("read")).status_code == 404
        assert api.get("/runs/" + snapshot["session_id"], headers=h("admin")).status_code == 200
        exposure = {
            "camera_id": "c2",
            "location_id": "L1",
            "start": query["start"],
            "end": query["end"],
            "road_users": 10,
            "method": "manual_count",
        }
        assert api.post("/exposures", json=exposure, headers=h("ingest")).status_code == 403
        assert api.post("/exposures", json=exposure, headers=h("admin")).status_code == 200
        assert not api.get("/stats", params=query, headers=h("read")).json()["exposure_normalized"].get("c2")


def test_disabled_credential_and_shared_key_do_not_bypass_scoped_mode(tmp_path):
    cfg, tokens = scoped_settings(tmp_path)
    cfg.api_key = "unused-legacy-key-with-enough-characters"
    cfg.service_credentials[0].enabled = False
    with TestClient(create_app(cfg)) as api:
        for token in [tokens["admin"], cfg.api_key, "wrong"]:
            assert api.get("/events", headers={"X-API-Key": token}).status_code == 401


def test_migration_adoption_upgrade_downgrade_and_backup_restore(tmp_path):
    cfg, tokens = scoped_settings(tmp_path)
    engine, factory = connect(cfg.database_url)
    assert migrate(engine, 1) == 1
    with pytest.raises(ValueError, match="mismatch"):
        require_current(engine)
    assert migrate(engine) == 2 and migrate(engine) == 2
    with TestClient(create_app(cfg)) as api:
        payload = event()
        assert api.post("/events", json=payload, headers={"X-API-Key": tokens["admin"]}).status_code == 200
    path = tmp_path / "backup.json"
    result = backup(engine, path)
    assert result["rows"] >= 2 and path.stat().st_mode & 0o777 == 0o600
    target, target_factory = connect(f"sqlite:///{tmp_path}/restore.db")
    assert restore(target, path)["restored_rows"] == result["rows"]
    with target_factory() as session:
        row = session.get(EventRow, payload["event_id"])
        assert row.payload["risk_score"] == payload["risk_score"]
    with pytest.raises(ValueError, match="empty target"):
        restore(target, path)
    tampered = json.loads(path.read_text())
    tampered["content"]["tables"]["events"][0]["payload"]["risk_score"] = 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="checksum"):
        restore(target, bad)
    with pytest.raises(ValueError, match="Downgrade"):
        migrate(target, 1)
    assert migrate(target, 1, allow_data_loss=True) == 1
    assert migrate(target) == 2
    with target_factory() as session:
        assert session.get(EventRow, payload["event_id"]) is not None
    engine.dispose()
    target.dispose()


def test_transient_failure_recovers_without_manual_restart(tmp_path):
    class Client:
        calls = 0
        ids = []

        def submit(self, value):
            self.calls += 1
            if self.calls <= 2:
                raise httpx.ConnectError("controlled outage")
            self.ids.append(value.event_id)

    client = Client()
    box = DeliveryOutbox(tmp_path / "recover.sqlite", client, retry_initial_s=0.01, retry_max_s=0.02)
    original = TrafficEvent.model_validate(event())
    box.enqueue(original)
    result = box.finish(2)
    assert result["sent"] == 1 and result["pending"] == 0 and result["failed_batches"] == 2
    assert client.ids == [original.event_id]


def test_permanent_rejection_is_retained_without_retry_storm(tmp_path):
    class Client:
        def submit(self, value):
            request = httpx.Request("POST", "http://localhost/events")
            response = httpx.Response(403, request=request)
            raise httpx.HTTPStatusError("controlled denial", request=request, response=response)

    box = DeliveryOutbox(tmp_path / "blocked.sqlite", Client())
    box.enqueue(TrafficEvent.model_validate(event()))
    result = box.finish(2)
    assert result["pending"] == result["blocked"] == 1 and result["delivery_batches_attempted"] == 1


def test_scheduled_retention_runs_after_startup(tmp_path):
    cfg, tokens = scoped_settings(tmp_path)
    cfg.retention_interval_s = 0.02
    app = create_app(cfg)
    with TestClient(app) as api:
        with app.state.session_factory() as session:
            row = EventRow(
                event_id=str(uuid4()),
                camera_id="c1",
                location_id="L1",
                timestamp=datetime.now(timezone.utc) - timedelta(days=90),
                synthetic=True,
                payload={},
            )
            session.add(row)
            session.commit()
        deadline = time.monotonic() + 2
        deleted = False
        while time.monotonic() < deadline:
            with app.state.session_factory() as session:
                deleted = session.scalar(select(EventRow)) is None
            if deleted:
                break
            time.sleep(0.02)
        assert deleted
        assert (
            api.get("/operations", headers={"X-API-Key": tokens["admin"]}).json()["retention"]["status"]
            == "ok"
        )


def test_compatible_legacy_tables_are_adopted_without_dropping_data(tmp_path):
    from roadsense.cloud.database import ExposureRow, RunRow

    engine, factory = connect(f"sqlite:///{tmp_path}/legacy.db")
    for model in [EventRow, ExposureRow, RunRow]:
        model.__table__.create(engine)
    original = event()
    with factory() as session:
        session.add(
            EventRow(
                event_id=original["event_id"],
                camera_id="c1",
                location_id="L1",
                timestamp=datetime.now(timezone.utc),
                synthetic=False,
                payload=original,
            )
        )
        session.commit()
    assert migrate(engine) == require_current(engine) == 2
    with factory() as session:
        assert session.get(EventRow, original["event_id"]).payload == original
    engine.dispose()


def test_legacy_outbox_schema_is_migrated_and_recovered(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.sqlite"
    original = TrafficEvent.model_validate(event())
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE pending (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        db.execute("INSERT INTO pending VALUES (?,?)", (str(original.event_id), original.model_dump_json()))
    received = []

    class Client:
        def submit(self, value):
            received.append(value.event_id)

    box = DeliveryOutbox(path, Client())
    snapshot = box.finish(2)
    assert snapshot["pending"] == 0 and received == [original.event_id]
    with sqlite3.connect(path) as db:
        assert {"created_at", "next_attempt", "state"} <= {
            row[1] for row in db.execute("PRAGMA table_info(pending)")
        }
