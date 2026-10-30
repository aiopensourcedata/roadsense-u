# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded event queries and idempotent event insertion."""

import hashlib
from datetime import datetime, timezone

from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError

from roadsense.cloud.database import AuditRow, EventRow, ExposureRow, RunRow
from roadsense.cloud.schemas import Exposure, TrafficEvent


def insert_event(session, event: TrafficEvent, *, retry_run_race: bool = True) -> bool:
    payload = event.model_dump(mode="json")
    # Ingestion is server time, independent of supplied capture/placeholder time.
    payload["ingested_at"] = datetime.now(timezone.utc).isoformat()
    identity_payload = {k: v for k, v in payload.items() if k != "ingested_at"}
    existing = session.get(EventRow, str(event.event_id))
    if existing:
        if (
            TrafficEvent.model_validate(existing.payload).model_dump(mode="json", exclude={"ingested_at"})
            != identity_payload
        ):
            raise ValueError("Event ID already exists with different content")
        return False
    if event.run_manifest is not None:
        run_id = str(event.session_id)
        manifest = event.run_manifest.model_dump(mode="json")
        run = session.get(RunRow, run_id)
        if run is not None and run.payload != manifest:
            raise ValueError("Run ID already exists with different manifest")
        if run is None:
            session.add(
                RunRow(session_id=run_id, manifest_sha256=event.run_manifest.sha256, payload=manifest)
            )
    session.add(
        EventRow(
            event_id=str(event.event_id),
            camera_id=event.camera_id,
            location_id=event.location_id,
            timestamp=event.timestamp,
            synthetic=event.synthetic,
            payload=payload,
        )
    )
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = session.get(EventRow, str(event.event_id))
        if existing is None and event.run_manifest is not None and retry_run_race:
            concurrent_run = session.get(RunRow, str(event.session_id))
            if concurrent_run is not None and concurrent_run.payload == event.run_manifest.model_dump(
                mode="json"
            ):
                session.rollback()
                return insert_event(session, event, retry_run_race=False)
        if (
            existing is None
            or TrafficEvent.model_validate(existing.payload).model_dump(mode="json", exclude={"ingested_at"})
            != identity_payload
        ):
            raise ValueError("Concurrent conflicting event ID")
        return False
    return True


def event_query(camera_id=None, location_id=None, start=None, end=None, synthetic=False):
    query = select(EventRow).where(EventRow.synthetic == synthetic)
    for column, value in [(EventRow.camera_id, camera_id), (EventRow.location_id, location_id)]:
        if value is not None:
            query = query.where(column == value)
    if start:
        query = query.where(EventRow.timestamp >= start)
    if end:
        query = query.where(EventRow.timestamp < end)
    return query.order_by(EventRow.timestamp, EventRow.event_id)


def insert_exposure(session, exposure: Exposure):
    # Serialize each camera's overlap check and write in one transaction.
    if session.bind.dialect.name == "postgresql":
        camera_key = f"{exposure.camera_id}:{exposure.synthetic}".encode()
        lock_id = int.from_bytes(hashlib.sha256(camera_key).digest()[:8], "big", signed=True)
        session.execute(text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": lock_id})
    elif session.bind.dialect.name == "sqlite":
        session.execute(text("BEGIN IMMEDIATE"))
    existing = session.get(ExposureRow, str(exposure.exposure_id))
    if existing:
        if existing.payload != exposure.model_dump(mode="json"):
            raise ValueError("Exposure ID conflict")
        return
    # Overlapping exposure windows would double-count denominators.
    overlap = session.scalar(
        select(ExposureRow).where(
            ExposureRow.camera_id == exposure.camera_id,
            ExposureRow.synthetic == exposure.synthetic,
            ExposureRow.start < exposure.end,
            ExposureRow.end > exposure.start,
        )
    )
    if overlap:
        raise ValueError("Overlapping camera exposure windows are not allowed")
    session.add(
        ExposureRow(
            exposure_id=str(exposure.exposure_id),
            camera_id=exposure.camera_id,
            location_id=exposure.location_id,
            start=exposure.start,
            end=exposure.end,
            road_users=exposure.road_users,
            synthetic=exposure.synthetic,
            payload=exposure.model_dump(mode="json"),
        )
    )
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = session.get(ExposureRow, str(exposure.exposure_id))
        if existing is None or existing.payload != exposure.model_dump(mode="json"):
            raise ValueError("Concurrent conflicting exposure ID")


def purge_before(session, cutoff: datetime):
    events = session.execute(delete(EventRow).where(EventRow.timestamp < cutoff)).rowcount
    exposure = session.execute(delete(ExposureRow).where(ExposureRow.end < cutoff)).rowcount
    session.execute(delete(AuditRow).where(AuditRow.timestamp < cutoff))
    session.execute(
        delete(RunRow).where(
            ~select(EventRow.event_id)
            .where(EventRow.payload["session_id"].as_string() == RunRow.session_id)
            .exists()
        )
    )
    session.commit()
    return {"events_deleted": events, "exposures_deleted": exposure}
