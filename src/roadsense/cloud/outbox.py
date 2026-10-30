# SPDX-License-Identifier: AGPL-3.0-only
"""Durable bounded delivery with persistent backoff, rejection and recovery state."""

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from pydantic import ValidationError

from roadsense.cloud.schemas import TrafficEvent


class DeliveryOutbox:
    """One producer/sender per file; delivery is at least once, UUID-idempotent.

    Transient failures retry with bounded backoff, including after restart.
    Permanent HTTP rejections remain blocked for explicit operator recovery.
    Full queues reject enqueues; the pipeline retains original JSONL for replay.
    """

    def __init__(
        self,
        path: Path,
        client,
        capacity: int = 1000,
        *,
        recover: bool = True,
        retry_initial_s: float = 0.2,
        retry_max_s: float = 5.0,
    ):
        if capacity < 1 or not 0 < retry_initial_s <= retry_max_s <= 3600:
            raise ValueError("Positive capacity and bounded retry delays required")
        self.path, self.client, self.capacity = path, client, capacity
        self.recover, self.retry_initial_s, self.retry_max_s = recover, retry_initial_s, retry_max_s
        self.lock = threading.Lock()
        self.wake, self.stop = threading.Event(), threading.Event()
        self.closed = False
        self.delivery_ms: list[float] = []
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS pending (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            columns = {r[1] for r in db.execute("PRAGMA table_info(pending)")}
            for name, definition in {
                "created_at": "REAL NOT NULL DEFAULT 0",
                "attempts": "INTEGER NOT NULL DEFAULT 0",
                "next_attempt": "REAL NOT NULL DEFAULT 0",
                "state": "TEXT NOT NULL DEFAULT 'pending'",
                "last_error": "TEXT",
            }.items():
                if name not in columns:
                    db.execute(
                        f"ALTER TABLE pending ADD COLUMN {name} {definition}"
                    )  # fixed internal definitions
            db.execute("UPDATE pending SET created_at=? WHERE created_at=0", (time.time(),))
            db.execute("CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        self.thread = threading.Thread(target=self._send, name="roadsense-delivery", daemon=True)
        self.thread.start()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=1)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _increment(db, name: str) -> None:
        db.execute("INSERT INTO counters VALUES (?,1) ON CONFLICT(name) DO UPDATE SET value=value+1", (name,))

    def enqueue(self, event: TrafficEvent) -> bool:
        body = event.model_dump_json()
        with self.lock, self._connect() as db:
            if self.closed:
                raise RuntimeError("Outbox is finalized")
            existing = db.execute("SELECT body FROM pending WHERE id=?", (str(event.event_id),)).fetchone()
            if existing is not None:
                if existing[0] != body:
                    raise ValueError("Outbox UUID already has different content")
                return True
            if db.execute("SELECT count(*) FROM pending").fetchone()[0] >= self.capacity:
                self._increment(db, "queue_rejected")
                return False
            db.execute(
                "INSERT INTO pending (id,body,created_at) VALUES (?,?,?)",
                (str(event.event_id), body, time.time()),
            )
        self.wake.set()
        return True

    def _send(self) -> None:
        while not self.stop.is_set():
            with self.lock, self._connect() as db:
                row = db.execute(
                    "SELECT * FROM pending WHERE state='pending' AND next_attempt<=? ORDER BY rowid LIMIT 1",
                    (time.time(),),
                ).fetchone()
                if row is not None:
                    db.execute("UPDATE pending SET attempts=attempts+1 WHERE id=?", (row["id"],))
                    self._increment(db, "delivery_batches_attempted")
            if row is None:
                self.wake.wait(0.05)
                self.wake.clear()
                continue
            started = time.perf_counter()
            try:
                self.client.submit(TrafficEvent.model_validate_json(row["body"]))
            except Exception as exc:
                permanent = isinstance(exc, ValidationError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and 400 <= exc.response.status_code < 500
                    and exc.response.status_code != 429
                )
                delay = min(self.retry_max_s, self.retry_initial_s * 2 ** min(row["attempts"], 20))
                with self.lock, self._connect() as db:
                    self._increment(db, "failed_batches")
                    db.execute(
                        "UPDATE pending SET state=?,next_attempt=?,last_error=? WHERE id=?",
                        (
                            "blocked" if permanent else "pending",
                            time.time() + delay,
                            type(exc).__name__,
                            row["id"],
                        ),
                    )
                if not self.recover:
                    break
            else:
                with self.lock, self._connect() as db:
                    db.execute("DELETE FROM pending WHERE id=?", (row["id"],))
                    self._increment(db, "sent")
                self.delivery_ms.append(1000 * (time.perf_counter() - started))
                self.delivery_ms = self.delivery_ms[-10000:]

    def snapshot(self) -> dict:
        with self.lock, self._connect() as db:
            rows = db.execute(
                "SELECT state,count(*) AS n,min(created_at) AS oldest FROM pending GROUP BY state"
            ).fetchall()
            counters = {r["name"]: r["value"] for r in db.execute("SELECT * FROM counters")}
        counts = {row["state"]: row["n"] for row in rows}
        oldest = min((row["oldest"] for row in rows), default=None)
        return {
            "sent": counters.get("sent", 0),
            "failed_batches": counters.get("failed_batches", 0),
            "delivery_batches_attempted": counters.get("delivery_batches_attempted", 0),
            "queue_rejected": counters.get("queue_rejected", 0),
            "pending": sum(counts.values()),
            "retryable_pending": counts.get("pending", 0),
            "blocked": counts.get("blocked", 0),
            "oldest_retained_age_s": max(0, time.time() - oldest) if oldest is not None else None,
            "worker_active": self.thread.is_alive(),
            "recover_automatically": self.recover,
            "delivery_latency_ms_recent": list(self.delivery_ms),
            "snapshot_semantics": "persistent cumulative counters; an in-flight request may finish after snapshot",
        }

    def finish(self, timeout_s: float = 1.0) -> dict:
        deadline = time.monotonic() + timeout_s
        while self.thread.is_alive() and time.monotonic() < deadline:
            if not self.snapshot()["retryable_pending"]:
                break
            self.thread.join(min(0.02, max(0, deadline - time.monotonic())))
        with self.lock:
            self.closed = True
        self.stop.set()
        self.wake.set()
        self.thread.join(max(0, deadline - time.monotonic()))
        return self.snapshot()
