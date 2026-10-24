# SPDX-License-Identifier: AGPL-3.0-only
"""Generate computed events, submit over HTTP, and independently verify PostgreSQL rows."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from generate_synthetic_data import generate
from sqlalchemy import select

from roadsense.cloud.client import EventClient
from roadsense.cloud.database import EventRow, connect
from roadsense.config import PipelineConfig, Settings
from roadsense.pipeline import run

if __name__ == "__main__":
    cfg = Settings()
    if not cfg.database_url.startswith("postgresql"):
        raise SystemExit("This integration check requires an actual PostgreSQL URL")
    with tempfile.TemporaryDirectory(prefix="roadsense-integration-") as directory:
        generate(str(Path(directory) / "input"))
        client = EventClient(cfg.api_url, cfg.api_key)
        try:
            report = run(
                str(Path(directory) / "input/crossing.avi"),
                str(Path(directory) / "run"),
                PipelineConfig(detector="synthetic", synthetic=True, camera_id="integration-fixture"),
                client,
            )
        finally:
            client.close()
        assert report["events"] >= 1 and report["submission_failures"] == 0
        engine, factory = connect(cfg.database_url)
        try:
            with factory() as session:
                rows = list(
                    session.scalars(select(EventRow).where(EventRow.camera_id == "integration-fixture"))
                )
                rows = [r for r in rows if r.payload["session_id"] == report["session_id"]]
                assert len(rows) == report["events"]
                assert rows[0].payload["features"]["cpa_distance"] < 5
            with httpx.Client(
                trust_env=False, base_url=cfg.api_url, headers={"X-API-Key": cfg.api_key}
            ) as http:
                response = http.get("/events", params={"synthetic": True, "camera_id": "integration-fixture"})
                response.raise_for_status()
                assert any(e["session_id"] == report["session_id"] for e in response.json())
                now = datetime.now(timezone.utc)
                response = http.get(
                    "/stats",
                    params={
                        "synthetic": True,
                        "start": (now - timedelta(days=1)).isoformat(),
                        "end": (now + timedelta(days=1)).isoformat(),
                        "camera_id": "integration-fixture",
                    },
                )
                response.raise_for_status()
                assert response.json()["event_count"] >= report["events"]
            evidence = {
                **report,
                "postgresql_independent_query": True,
                "http_retrieval": True,
                "http_statistics": True,
            }
            output = Path("outputs/postgres-integration")
            output.mkdir(parents=True, exist_ok=True)
            (output / "verification.json").write_text(json.dumps(evidence, indent=2))
            print(
                json.dumps(
                    {
                        "computed_events": report["events"],
                        "postgresql_verified": True,
                        "body_bytes": report["network_body_bytes_attempted"],
                    }
                )
            )
        finally:
            engine.dispose()
