# SPDX-License-Identifier: AGPL-3.0-only
"""Controlled actual-HTTP delivery experiments; simulated senders, not camera accuracy."""

import argparse
import asyncio
import copy
import json
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
import numpy as np
import psutil
import uvicorn
from fastapi.responses import JSONResponse
from sqlalchemy import select

from roadsense.cloud.api import create_app
from roadsense.cloud.client import EventClient
from roadsense.cloud.database import EventRow
from roadsense.cloud.outbox import DeliveryOutbox
from roadsense.cloud.schemas import TrafficEvent
from roadsense.config import PipelineConfig, Settings
from roadsense.pipeline import run
from roadsense.provenance import canonical_digest, file_sha256


def benchmark(output: Path, cameras: list[int], repeats: int, events: int) -> dict:
    if output.exists() or not cameras or min(cameras) < 1 or max(cameras) > 16 or repeats < 2 or events < 1:
        raise ValueError("Fresh output, 1–16 simulated senders, >=2 trials and positive events required")
    output.mkdir(parents=True)
    subprocess.run(
        [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(output / "input")],
        check=True,
        capture_output=True,
    )
    run(
        str(output / "input/crossing.avi"),
        str(output / "computed"),
        PipelineConfig(detector="synthetic", synthetic=True),
    )
    source = output / "computed/events.jsonl"
    template = json.loads(source.read_text().splitlines()[0])
    profiles = ["healthy", "delay", "outage", "rate_limit", "lost_ack", "queue_full"]
    modes = ["synchronous", "stop_after_failure", "recoverable"]
    plan = {
        "schema_version": 1,
        "synthetic": True,
        "seed": 42,
        "cameras": cameras,
        "repeats": repeats,
        "events_per_camera": events,
        "profiles": profiles,
        "modes": modes,
        "emission_interval_s": 0.01,
        "fault_duration_s": 0.12,
        "delay_s": 0.02,
        "illustrative_ack_deadline_s": 0.5,
        "queue_full_capacity": 1,
        "queue_full_emission_interval_s": 0.001,
        "study_status": "development-only; budgets are illustrative, not preregistered operational targets",
        "drain_budget_s": 2,
        "queue_capacity_per_camera": max(4, events),
        "http_attempts_per_batch": 1,
        "retry_initial_s": 0.01,
        "retry_max_s": 0.05,
        "template_sha256": file_sha256(source),
        "code_sha256": file_sha256(Path(__file__)),
        "scope": "Frozen computed synthetic event features, simulated sender load, actual loopback HTTP/SQLite; no detector capacity, WAN, labeled warning or cloud-price claim",
    }
    (output / "frozen-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    key = uuid5(NAMESPACE_URL, str(output.resolve())).hex
    app = create_app(
        Settings(
            api_key=key, auth_mode="development_shared_key", database_url=f"sqlite:///{output}/service.db"
        )
    )
    control = {"profile": "healthy", "start": 0.0, "seen": set()}

    @app.middleware("http")
    async def faults(request, call_next):
        if request.method != "POST" or request.url.path != "/events":
            return await call_next(request)
        age = time.monotonic() - control["start"]
        if control["profile"] in {"outage", "rate_limit"} and age < plan["fault_duration_s"]:
            return JSONResponse(
                {"detail": "controlled experiment fault"},
                status_code=503 if control["profile"] == "outage" else 429,
            )
        if control["profile"] in {"delay", "queue_full"}:
            await asyncio.sleep(plan["delay_s"])
        event_id = (await request.json())["event_id"]
        response = await call_next(request)
        if control["profile"] == "lost_ack" and event_id not in control["seen"]:
            control["seen"].add(event_id)
            return JSONResponse({"detail": "controlled lost acknowledgement"}, status_code=503)
        return response

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    base = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        thread.join(0.01)
    if not server.started:
        raise RuntimeError("Benchmark service did not start")
    results = []
    try:
        for count in cameras:
            for profile in profiles:
                for repetition in range(repeats):
                    # Rotate execution order to reduce fixed ordering/thermal bias.
                    order = modes[repetition % 3 :] + modes[: repetition % 3]
                    for mode in order:
                        trial = f"{count}-{profile}-{repetition}-{mode}"
                        trial_dir = output / trial
                        trial_dir.mkdir()
                        expected, acks, ages, production_lag = [], set(), [], []
                        clients, boxes = [], []
                        control.update(profile=profile, start=float("inf"), seen=set())
                        for camera in range(count):
                            raw = EventClient(base, key, attempts=1)

                            class MeasuredClient:
                                def __init__(self, client):
                                    self.client = client

                                def submit(self, value):
                                    response = self.client.submit(value)
                                    acks.add(str(value.event_id))
                                    ages.append(time.monotonic() - created[str(value.event_id)])
                                    return response

                            clients.append(raw)
                            if mode != "synchronous":
                                boxes.append(
                                    DeliveryOutbox(
                                        trial_dir / f"camera-{camera}.sqlite",
                                        MeasuredClient(raw),
                                        capacity=1
                                        if profile == "queue_full"
                                        else plan["queue_capacity_per_camera"],
                                        recover=mode == "recoverable",
                                        retry_initial_s=plan["retry_initial_s"],
                                        retry_max_s=plan["retry_max_s"],
                                    )
                                )
                        created = {}
                        cpu_started = time.process_time()
                        start = time.monotonic()
                        control["start"] = start

                        def produce(camera):
                            camera_id = f"load-{trial}-{camera}"
                            session = str(uuid5(NAMESPACE_URL, trial + str(camera)))
                            for k in range(events):
                                scheduled = start + k * (
                                    plan["queue_full_emission_interval_s"]
                                    if profile == "queue_full"
                                    else plan["emission_interval_s"]
                                )
                                time.sleep(max(0, scheduled - time.monotonic()))
                                body = copy.deepcopy(template)
                                body.update(
                                    event_id=str(uuid5(NAMESPACE_URL, f"{trial}/{camera}/{k}")),
                                    session_id=session,
                                    camera_id=camera_id,
                                )
                                manifest = body["run_manifest"]["content"]
                                manifest.update(session_id=session, camera_id=camera_id)
                                manifest["config"]["camera_id"] = camera_id
                                body["run_manifest"]["sha256"] = canonical_digest(manifest)
                                value = TrafficEvent.model_validate(body)
                                expected.append(str(value.event_id))
                                created[str(value.event_id)] = scheduled
                                production_lag.append(max(0, time.monotonic() - scheduled))
                                if mode == "synchronous":
                                    try:
                                        clients[camera].submit(value)
                                        acks.add(str(value.event_id))
                                        ages.append(time.monotonic() - scheduled)
                                    except httpx.HTTPError:
                                        pass
                                else:
                                    boxes[camera].enqueue(value)

                        producers = [
                            threading.Thread(target=produce, args=(camera,)) for camera in range(count)
                        ]
                        for producer in producers:
                            producer.start()
                        for producer in producers:
                            producer.join()
                        assert len(expected) == count * events, (
                            "Producer failure must not yield a success report"
                        )
                        snapshots = [box.finish(plan["drain_budget_s"]) for box in boxes]
                        assert not any(snapshot["worker_active"] for snapshot in snapshots), (
                            "Unsettled trial; choose a separate longer-budget experiment"
                        )
                        elapsed = time.monotonic() - start
                        with app.state.session_factory() as session:
                            persisted = set(
                                session.scalars(
                                    select(EventRow.event_id).where(EventRow.event_id.in_(expected))
                                )
                            )
                        attempted = sum(client.bytes_attempted for client in clients)
                        for client in clients:
                            client.close()
                        row = {
                            "trial": trial,
                            "mode": mode,
                            "profile": profile,
                            "simulated_senders": count,
                            "repetition": repetition,
                            "generated": len(expected),
                            "persisted_unique": len(persisted),
                            "acknowledged_unique": len(acks),
                            "persisted_fraction": len(persisted) / len(expected),
                            "unpersisted_ids": sorted(set(expected) - persisted),
                            "persisted_but_unacknowledged_ids": sorted(persisted - acks),
                            "timely_ack_fraction": sum(
                                age <= plan["illustrative_ack_deadline_s"] for age in ages
                            )
                            / len(expected),
                            "ack_ages_s": list(ages),
                            "cpu_time_s_same_process_client_and_server": time.process_time() - cpu_started,
                            "rss_end_mb": psutil.Process().memory_info().rss / 2**20,
                            "pending": sum(s["pending"] for s in snapshots),
                            "rejected": sum(s["queue_rejected"] for s in snapshots),
                            "body_bytes_attempted": attempted,
                            "elapsed_s": elapsed,
                            "producer_lag_p95_ms": float(np.percentile(production_lag, 95) * 1000),
                            "ack_age_p50_ms": float(np.percentile(ages, 50) * 1000) if ages else None,
                            "ack_age_p99_ms": float(np.percentile(ages, 99) * 1000) if ages else None,
                            "ack_age_p95_ms": float(np.percentile(ages, 95) * 1000) if ages else None,
                            "generated_ids": sorted(expected),
                            "persisted_ids": sorted(persisted),
                            "scope": "synthetic repeated computed metadata; not independent traffic events",
                        }
                        results.append(row)
                        (trial_dir / "result.json").write_text(
                            json.dumps(row, indent=2, allow_nan=False) + "\n"
                        )
        summary = []
        rng = np.random.default_rng(42)
        for count in cameras:
            for profile in profiles:
                for mode in modes:
                    selected = [
                        r
                        for r in results
                        if (r["simulated_senders"], r["profile"], r["mode"]) == (count, profile, mode)
                    ]
                    values = np.array([r["persisted_fraction"] for r in selected])
                    bootstrap = np.mean(rng.choice(values, (1000, len(values)), replace=True), axis=1)
                    summary.append(
                        {
                            "simulated_senders": count,
                            "profile": profile,
                            "mode": mode,
                            "trials": len(values),
                            "mean_persisted_fraction": float(values.mean()),
                            "run_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                            "median_producer_lag_p95_ms": statistics.median(
                                r["producer_lag_p95_ms"] for r in selected
                            ),
                            "mean_attempted_body_bytes": statistics.mean(
                                r["body_bytes_attempted"] for r in selected
                            ),
                        }
                    )
        report = {
            "plan": plan,
            "trials": len(results),
            "summary": summary,
            "limitations": [
                "Few repeated local runs; percentile intervals may degenerate and are descriptive.",
                "SQLite loopback workload, synthetic metadata and simulated senders do not establish WAN/cloud or detector scaling.",
                "Retry recovery consumes additional attempts/bytes; compare measured costs, not only eventual persistence.",
                "Capture time, traffic accuracy, paid-cloud cost and SLA compliance are not evaluated.",
            ],
        }
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        server.should_exit = True
        thread.join(10)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cameras", type=int, nargs="+", default=[1, 4])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--events", type=int, default=6)
    a = p.parse_args()
    report = benchmark(a.output, a.cameras, a.repeats, a.events)
    print(json.dumps({"trials": report["trials"], "output": str(a.output), "synthetic": True}))
