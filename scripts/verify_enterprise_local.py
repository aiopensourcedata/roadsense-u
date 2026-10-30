# SPDX-License-Identifier: AGPL-3.0-only
"""Verify local TLS, scoped ingestion, PostgreSQL backup/restore and schema recovery."""

import argparse
import copy
import json
import ssl
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from sqlalchemy import create_engine, func, select, table
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateSchema, DropSchema

from roadsense.cloud.backup import backup, restore
from roadsense.cloud.client import EventClient
from roadsense.cloud.migrations import migrate, require_current
from roadsense.config import PipelineConfig, Settings
from roadsense.pipeline import run


def verify(output: Path, credentials: Path, ca: Path, base: str) -> dict:
    if (
        output.exists()
        or urlparse(base).hostname not in {"localhost", "127.0.0.1"}
        or not base.startswith("https://")
    ):
        raise ValueError("Fresh output and loopback HTTPS target required")
    output.mkdir(parents=True)
    tokens = {
        role: dotenv_values(credentials / f"{role}.env")["ROADSENSE_API_KEY"]
        for role in ["admin", "read", "ingest"]
    }
    context = ssl.create_default_context(cafile=str(ca))
    with httpx.Client(base_url=base, verify=context, trust_env=False, timeout=10) as http:
        assert http.get("/health").status_code == 200
        assert http.get("/events").status_code == 401
        subprocess.run(
            [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(output / "input")],
            check=True,
            capture_output=True,
        )
        sender = EventClient(base, tokens["ingest"], ca_file=str(ca))
        try:
            report = run(
                str(output / "input/crossing.avi"),
                str(output / "run"),
                PipelineConfig(detector="synthetic", synthetic=True, delivery_shutdown_s=5),
                client=sender,
            )
        finally:
            sender.close()
        assert report["events"] >= 1 and report["delivery"]["pending"] == 0

        def headers(role):
            return {"X-API-Key": tokens[role]}

        rows = http.get(
            "/events", params={"synthetic": True, "camera_id": "camera-local"}, headers=headers("read")
        ).json()
        ids = {row["event_id"] for row in rows}
        emitted = [json.loads(line) for line in (output / "run/events.jsonl").read_text().splitlines()]
        assert {row["event_id"] for row in emitted} <= ids
        assert http.get("/events", headers=headers("ingest")).status_code == 403
        assert http.get("/events?camera_id=forbidden-camera", headers=headers("read")).status_code == 403
        assert http.get("/operations", headers=headers("read")).status_code == 403
        assert http.get("/operations", headers=headers("admin")).json()["schema_revision"] == 2
        bad = copy.deepcopy(emitted[0])
        bad["event_id"] = str(uuid4())
        assert http.post("/events", json=bad, headers=headers("read")).status_code == 403
        assert http.post("/events", json=emitted[0], headers=headers("ingest")).json()["created"] is False
        assert http.get("/runs/" + report["session_id"], headers=headers("read")).status_code == 200
        # Concurrent distinct events referring to a new shared run exercise manifest insertion races.
        concurrent = []
        from roadsense.provenance import canonical_digest

        session_id = str(uuid4())
        for _ in range(4):
            value = copy.deepcopy(emitted[0])
            value["event_id"] = str(uuid4())
            value["session_id"] = session_id
            value["run_manifest"]["content"]["session_id"] = session_id
            value["run_manifest"]["sha256"] = canonical_digest(value["run_manifest"]["content"])
            concurrent.append(value)
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(
                pool.map(
                    lambda value: http.post("/events", json=value, headers=headers("ingest")).status_code,
                    concurrent,
                )
            )
        assert statuses == [200] * 4, f"Concurrent ingestion failed: {statuses}"
        audit = http.get("/audit", headers=headers("admin")).json()
        assert any(row["principal_id"] == "local-ingest" for row in audit)
    url = make_url(Settings().database_url).set(host="127.0.0.1", port=55432, database="roadsense_enterprise")
    engine = create_engine(url)
    snapshot = backup(engine, output / "metadata-backup.json")
    schema = "rs_restore_" + uuid4().hex[:16]
    target = None
    try:
        with engine.begin() as connection:
            connection.execute(CreateSchema(schema))
        target = create_engine(url.update_query_dict({"options": f"-csearch_path={schema}"}))
        restored = restore(target, output / "metadata-backup.json")
        assert restored["restored_rows"] == snapshot["rows"]
        with target.connect() as connection:
            restored_events = connection.scalar(select(func.count()).select_from(table("events")))
            restored_runs = connection.scalar(select(func.count()).select_from(table("runs")))
        assert restored_events >= report["events"] + 4 and restored_runs >= 2
        assert migrate(target, 1, allow_data_loss=True) == 1
        assert migrate(target) == require_current(target) == 2
    finally:
        if target:
            target.dispose()
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        engine.dispose()
    result = {
        "synthetic": True,
        "https_certificate_verified": True,
        "scoped_read_write_denials": True,
        "events_computed": report["events"],
        "concurrent_new_run_ingests": 4,
        "backup_rows": snapshot["rows"],
        "backup_restored_and_counts_verified": True,
        "isolated_schema_downgrade_upgrade": True,
        "source_database_preserved": True,
        "schema_revision": 2,
        "delivery_pending": 0,
        "scope": "Local software integration; no independent traffic accuracy or remote/cloud SLA validation",
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--credentials", type=Path, default=Path("outputs/cloud-enterprise/credentials"))
    p.add_argument("--ca", type=Path, default=Path("outputs/cloud-enterprise/tls/root.crt"))
    p.add_argument("--base-url", default="https://localhost:18443")
    a = p.parse_args()
    verify(a.output, a.credentials, a.ca, a.base_url)
