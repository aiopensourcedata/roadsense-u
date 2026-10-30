# SPDX-License-Identifier: AGPL-3.0-only
"""Authenticated metadata service. TLS must terminate at the remote deployment ingress."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import AwareDatetime
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import SQLAlchemyError

from roadsense.analytics.aggregation import aggregate
from roadsense.cloud.auth import Principal, resolve_principal
from roadsense.cloud.database import AuditRow, EventRow, ExposureRow, RunRow, connect
from roadsense.cloud.migrations import CURRENT_REVISION, migrate, require_current
from roadsense.cloud.operations import Operations
from roadsense.cloud.repository import event_query, insert_event, insert_exposure, purge_before
from roadsense.cloud.schemas import Exposure, TrafficEvent
from roadsense.config import Settings

log = logging.getLogger("roadsense.api")


class BodyLimit:
    """Read at most limit bytes before JSON parsing, including chunked requests."""

    def __init__(self, app, limit: int, operations: Operations):
        self.app, self.limit, self.operations = app, limit, operations

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                self.operations.observe(413, 0)
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": b'{"detail":"Request too large"}'})
                return
            if message.get("body"):
                chunks.append(message["body"])
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        started = time.perf_counter()

        async def logged_send(message):
            if message["type"] == "http.response.start":
                self.operations.observe(message["status"], 1000 * (time.perf_counter() - started))
                log.info(
                    json.dumps(
                        {
                            "event": "http_request",
                            "method": scope["method"],
                            "path": scope["path"],
                            "status": message["status"],
                            "principal_id": scope.get("roadsense_principal"),
                            "duration_ms": round(1000 * (time.perf_counter() - started), 2),
                        }
                    )
                )
            await send(message)

        await self.app(scope, replay, logged_send)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    engine, factory = connect(cfg.database_url)

    operations = Operations()

    def run_retention():
        try:
            with factory() as s:
                result = purge_before(s, datetime.now(timezone.utc) - timedelta(days=cfg.retention_days))
            operations.retention = {"status": "ok", "at": datetime.now(timezone.utc).isoformat(), **result}
        except SQLAlchemyError:
            operations.retention = {"status": "failed", "at": datetime.now(timezone.utc).isoformat()}
            log.error(json.dumps({"event": "retention_failed"}))
            raise

    async def retention_loop():
        while True:
            await asyncio.sleep(cfg.retention_interval_s)
            try:
                await asyncio.to_thread(run_retention)
            except SQLAlchemyError:
                pass  # visible failure state; next scheduled pass retries

    @asynccontextmanager
    async def lifespan(app):
        if cfg.auto_migrate:
            migrate(engine)
        else:
            require_current(engine)
        run_retention()
        task = asyncio.create_task(retention_loop()) if cfg.retention_interval_s else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            engine.dispose()

    app = FastAPI(title="RoadSense-U research API", version="0.1.0", lifespan=lifespan)
    app.state.engine, app.state.session_factory = engine, factory
    app.add_middleware(BodyLimit, limit=cfg.max_request_bytes, operations=operations)

    def authenticate(request: Request, x_api_key: str = Header(default="")) -> Principal:
        principal = resolve_principal(cfg, x_api_key)
        request.scope["roadsense_principal"] = principal.credential_id
        return principal

    def permission(role):
        def permitted(principal: Principal = Depends(authenticate)):
            principal.require_role(role)
            return principal

        return permitted

    def scoped(query, column, principal):
        return query if "*" in principal.cameras else query.where(column.in_(principal.cameras))

    def audit(s, principal, action, camera=None, resource=None):
        s.add(
            AuditRow(
                audit_id=str(uuid4()),
                timestamp=datetime.now(timezone.utc),
                principal_id=principal.credential_id,
                action=action,
                camera_id=camera,
                resource_id=resource,
            )
        )
        s.commit()

    def session():
        with factory() as s:
            yield s

    @app.get("/health")
    def health(s=Depends(session)):
        try:
            s.execute(text("SELECT 1"))
        except SQLAlchemyError:
            raise HTTPException(503, "Database unavailable")
        return {"status": "ok", "service": "roadsense-u", "research_only": True}

    @app.post("/events")
    def submit(event: TrafficEvent, principal=Depends(permission("ingest")), s=Depends(session)):
        principal.require_camera(event.camera_id)
        if event.timestamp < datetime.now(timezone.utc) - timedelta(days=cfg.retention_days):
            raise HTTPException(422, "Event is older than configured retention window")
        try:
            created = insert_event(s, event)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        audit(s, principal, "event_ingest", event.camera_id, str(event.event_id))
        return {"event_id": str(event.event_id), "created": created}

    @app.get("/events", response_model=list[TrafficEvent])
    def events(
        camera_id: str | None = None,
        location_id: str | None = None,
        start: AwareDatetime | None = None,
        end: AwareDatetime | None = None,
        synthetic: bool = False,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        principal=Depends(permission("read")),
        s=Depends(session),
    ):
        if camera_id:
            principal.require_camera(camera_id)
        if start and end and start >= end:
            raise HTTPException(422, "start must precede end")
        return [
            r.payload
            for r in s.scalars(
                scoped(
                    event_query(camera_id, location_id, start, end, synthetic), EventRow.camera_id, principal
                )
                .offset(offset)
                .limit(limit)
            )
        ]

    @app.get("/runs/{session_id}")
    def run_manifest(session_id: UUID, principal=Depends(permission("read")), s=Depends(session)):
        row = s.get(RunRow, str(session_id))
        if row is None or not principal.permits_camera(row.payload["content"]["camera_id"]):
            raise HTTPException(404, "Run manifest unavailable or outside scope")
        return row.payload

    @app.delete("/events/{event_id}")
    def remove(event_id: UUID, principal=Depends(permission("admin")), s=Depends(session)):
        row = s.get(EventRow, str(event_id))
        if row is not None and not principal.permits_camera(row.camera_id):
            raise HTTPException(404, "Event unavailable or outside scope")
        count = s.execute(delete(EventRow).where(EventRow.event_id == str(event_id))).rowcount
        s.commit()
        audit(s, principal, "event_delete", row.camera_id if row else None, str(event_id))
        return {"deleted": count}

    @app.post("/exposures")
    def exposure(exposure: Exposure, principal=Depends(permission("ingest")), s=Depends(session)):
        principal.require_camera(exposure.camera_id)
        if exposure.end < datetime.now(timezone.utc) - timedelta(days=cfg.retention_days):
            raise HTTPException(422, "Exposure is older than configured retention window")
        try:
            insert_exposure(s, exposure)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        audit(s, principal, "exposure_ingest", exposure.camera_id, str(exposure.exposure_id))
        return {"exposure_id": str(exposure.exposure_id)}

    @app.get("/stats")
    def stats(
        start: AwareDatetime,
        end: AwareDatetime,
        synthetic: bool = False,
        camera_id: str | None = None,
        location_id: str | None = None,
        principal=Depends(permission("read")),
        s=Depends(session),
    ):
        if camera_id:
            principal.require_camera(camera_id)
        if start >= end or end - start > timedelta(days=366):
            raise HTTPException(422, "Provide an ordered range of at most 366 days")
        rows = list(
            s.scalars(
                scoped(
                    event_query(camera_id, location_id, start, end, synthetic), EventRow.camera_id, principal
                ).limit(100001)
            )
        )
        if len(rows) > 100000:
            raise HTTPException(422, "Narrow the statistics time range")
        windows = select(ExposureRow).where(
            ExposureRow.synthetic == synthetic, ExposureRow.start >= start, ExposureRow.end <= end
        )
        windows = scoped(windows, ExposureRow.camera_id, principal)
        if camera_id:
            windows = windows.where(ExposureRow.camera_id == camera_id)
        if location_id:
            windows = windows.where(ExposureRow.location_id == location_id)
        return aggregate([r.payload for r in rows], [r.payload for r in s.scalars(windows)])

    @app.post("/maintenance/purge")
    def purge(principal=Depends(permission("admin")), s=Depends(session)):
        if "*" not in principal.cameras:
            raise HTTPException(403, "Global maintenance requires wildcard administrator scope")
        result = purge_before(s, datetime.now(timezone.utc) - timedelta(days=cfg.retention_days))
        audit(s, principal, "retention_purge")
        return result

    @app.get("/operations")
    def operational_metrics(principal=Depends(permission("admin")), s=Depends(session)):
        if "*" not in principal.cameras:
            raise HTTPException(403, "Global metrics require wildcard administrator scope")
        return {
            **operations.snapshot(),
            "schema_revision": CURRENT_REVISION,
            "auth_mode": cfg.auth_mode,
            "stored_events": s.scalar(select(func.count()).select_from(EventRow)),
            "stored_runs": s.scalar(select(func.count()).select_from(RunRow)),
            "retention_interval_s": cfg.retention_interval_s,
        }

    @app.get("/audit")
    def audit_log(
        limit: int = Query(100, ge=1, le=1000), principal=Depends(permission("admin")), s=Depends(session)
    ):
        query = (
            scoped(select(AuditRow), AuditRow.camera_id, principal)
            .order_by(AuditRow.timestamp.desc())
            .limit(limit)
        )
        return [
            {
                "timestamp": r.timestamp,
                "principal_id": r.principal_id,
                "action": r.action,
                "camera_id": r.camera_id,
                "resource_id": r.resource_id,
            }
            for r in s.scalars(query)
        ]

    return app
