# SPDX-License-Identifier: AGPL-3.0-only
"""Authenticated metadata service. TLS must terminate at the remote deployment ingress."""

import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import AwareDatetime
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError

from roadsense.analytics.aggregation import aggregate
from roadsense.cloud.database import Base, EventRow, ExposureRow, connect
from roadsense.cloud.repository import event_query, insert_event, insert_exposure, purge_before
from roadsense.cloud.schemas import Exposure, TrafficEvent
from roadsense.config import Settings

log = logging.getLogger("roadsense.api")


class BodyLimit:
    """Read at most limit bytes before JSON parsing, including chunked requests."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

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
                log.info(
                    json.dumps(
                        {
                            "event": "http_request",
                            "method": scope["method"],
                            "path": scope["path"],
                            "status": message["status"],
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

    @asynccontextmanager
    async def lifespan(app):
        Base.metadata.create_all(engine)
        with factory() as s:
            purge_before(s, datetime.now(timezone.utc) - timedelta(days=cfg.retention_days))
        yield
        engine.dispose()

    app = FastAPI(title="RoadSense-U research API", version="0.1.0", lifespan=lifespan)
    app.state.engine, app.state.session_factory = engine, factory
    app.add_middleware(BodyLimit, limit=cfg.max_request_bytes)

    def authenticate(x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key.encode(), cfg.api_key.encode()):
            raise HTTPException(401, "Invalid API key")

    def session():
        with factory() as s:
            yield s

    protected = [Depends(authenticate)]

    @app.get("/health")
    def health(s=Depends(session)):
        try:
            s.execute(text("SELECT 1"))
        except SQLAlchemyError:
            raise HTTPException(503, "Database unavailable")
        return {"status": "ok", "service": "roadsense-u", "research_only": True}

    @app.post("/events", dependencies=protected)
    def submit(event: TrafficEvent, s=Depends(session)):
        if event.timestamp < datetime.now(timezone.utc) - timedelta(days=cfg.retention_days):
            raise HTTPException(422, "Event is older than configured retention window")
        try:
            created = insert_event(s, event)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"event_id": str(event.event_id), "created": created}

    @app.get("/events", dependencies=protected, response_model=list[TrafficEvent])
    def events(
        camera_id: str | None = None,
        location_id: str | None = None,
        start: AwareDatetime | None = None,
        end: AwareDatetime | None = None,
        synthetic: bool = False,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        s=Depends(session),
    ):
        if start and end and start >= end:
            raise HTTPException(422, "start must precede end")
        return [
            r.payload
            for r in s.scalars(
                event_query(camera_id, location_id, start, end, synthetic).offset(offset).limit(limit)
            )
        ]

    @app.delete("/events/{event_id}", dependencies=protected)
    def remove(event_id: UUID, s=Depends(session)):
        count = s.execute(delete(EventRow).where(EventRow.event_id == str(event_id))).rowcount
        s.commit()
        return {"deleted": count}

    @app.post("/exposures", dependencies=protected)
    def exposure(exposure: Exposure, s=Depends(session)):
        if exposure.end < datetime.now(timezone.utc) - timedelta(days=cfg.retention_days):
            raise HTTPException(422, "Exposure is older than configured retention window")
        try:
            insert_exposure(s, exposure)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"exposure_id": str(exposure.exposure_id)}

    @app.get("/stats", dependencies=protected)
    def stats(
        start: AwareDatetime,
        end: AwareDatetime,
        synthetic: bool = False,
        camera_id: str | None = None,
        location_id: str | None = None,
        s=Depends(session),
    ):
        if start >= end or end - start > timedelta(days=366):
            raise HTTPException(422, "Provide an ordered range of at most 366 days")
        rows = list(s.scalars(event_query(camera_id, location_id, start, end, synthetic).limit(100001)))
        if len(rows) > 100000:
            raise HTTPException(422, "Narrow the statistics time range")
        windows = select(ExposureRow).where(
            ExposureRow.synthetic == synthetic, ExposureRow.start >= start, ExposureRow.end <= end
        )
        if camera_id:
            windows = windows.where(ExposureRow.camera_id == camera_id)
        if location_id:
            windows = windows.where(ExposureRow.location_id == location_id)
        return aggregate([r.payload for r in rows], [r.payload for r in s.scalars(windows)])

    @app.post("/maintenance/purge", dependencies=protected)
    def purge(s=Depends(session)):
        return purge_before(s, datetime.now(timezone.utc) - timedelta(days=cfg.retention_days))

    return app
