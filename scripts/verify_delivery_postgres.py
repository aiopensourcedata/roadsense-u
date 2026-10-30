# SPDX-License-Identifier: AGPL-3.0-only
"""Isolated synthetic video -> actual HTTP -> PostgreSQL delivery/provenance check."""

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from sqlalchemy import create_engine, func, select, table
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateSchema, DropSchema

from roadsense.cloud.api import create_app
from roadsense.cloud.client import EventClient
from roadsense.config import PipelineConfig, Settings
from roadsense.pipeline import run


def verify(output: Path):
    if output.exists():
        raise ValueError("Choose a fresh output directory")
    cfg = Settings()
    url = make_url(cfg.database_url)
    if url.get_backend_name() != "postgresql" or url.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("This check requires an explicitly configured local PostgreSQL database")
    schema = "rs_delivery_" + uuid4().hex[:16]
    engine = create_engine(url, connect_args={"connect_timeout": 5})
    created = False
    server = thread = None
    output.mkdir(parents=True)
    try:
        with engine.begin() as connection:
            connection.execute(CreateSchema(schema))
        created = True
        isolated = url.update_query_dict({"options": f"-csearch_path={schema}"})
        settings = Settings(api_key=cfg.api_key, database_url=isolated.render_as_string(hide_password=False))
        app = create_app(settings)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            thread.join(0.02)
        if not server.started:
            raise RuntimeError("Isolated API failed to start")
        base = f"http://127.0.0.1:{port}"
        subprocess.run(
            [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(output / "input")],
            check=True,
            capture_output=True,
        )
        client = EventClient(base, cfg.api_key)
        try:
            report = run(
                str(output / "input/crossing.avi"),
                str(output / "run"),
                PipelineConfig(detector="synthetic", synthetic=True, delivery_shutdown_s=5),
                client=client,
            )
        finally:
            client.close()
        headers = {"X-API-Key": cfg.api_key}
        with httpx.Client(base_url=base, trust_env=False, timeout=5) as http:
            assert http.get("/runs/" + report["session_id"]).status_code == 401
            response = http.get("/events?synthetic=true", headers=headers)
            response.raise_for_status()
            rows = response.json()
            response = http.get("/runs/" + report["session_id"], headers=headers)
            response.raise_for_status()
            manifest = response.json()
            assert report["delivery"]["pending"] == 0 and len(rows) == report["events"] >= 1
            assert manifest["sha256"] == report["run_manifest_sha256"]
            assert all(
                row["ingested_at"] and row["timestamp_origin"] == "ingestion_start_not_capture_time"
                for row in rows
            )
            first = json.loads((output / "run/events.jsonl").read_text().splitlines()[0])
            assert http.post("/events", json=first, headers=headers).json()["created"] is False
        with engine.connect() as connection:
            persisted = connection.scalar(select(func.count()).select_from(table("events", schema=schema)))
            runs = connection.scalar(select(func.count()).select_from(table("runs", schema=schema)))
        assert persisted == len(rows) and runs == 1
        record = {
            "synthetic": True,
            "actual_http": True,
            "database": "PostgreSQL isolated temporary schema",
            "stored_events": persisted,
            "stored_run_manifests": runs,
            "idempotent_replay": True,
            "manifest_binding_verified": True,
            "server_ingestion_separate": True,
            "delivery_pending": 0,
            "capture_age_available": False,
            "scope": "software integration; not labeled traffic validation",
        }
        (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(10)
        if created:
            with engine.begin() as connection:
                connection.execute(DropSchema(schema, cascade=True))
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.output)
    except Exception as exc:
        raise SystemExit(
            f"Isolated delivery verification failed: {type(exc).__name__}; check local PostgreSQL/API access and output artifacts"
        ) from None
