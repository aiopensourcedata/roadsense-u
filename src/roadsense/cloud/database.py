# SPDX-License-Identifier: AGPL-3.0-only
"""SQLAlchemy storage; PostgreSQL deployment and explicit SQLite test/development support."""

from sqlalchemy import JSON, Boolean, DateTime, Index, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class AuditRow(Base):
    __tablename__ = "audit_log"
    audit_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    timestamp: Mapped[object] = mapped_column(DateTime(timezone=True), index=True)
    principal_id: Mapped[str] = mapped_column(String(100), index=True)
    action: Mapped[str] = mapped_column(String(100))
    camera_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(100), nullable=True)


class RunRow(Base):
    __tablename__ = "runs"
    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)


class EventRow(Base):
    __tablename__ = "events"
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(100), index=True)
    location_id: Mapped[str] = mapped_column(String(100), index=True)
    timestamp: Mapped[object] = mapped_column(DateTime(timezone=True), index=True)
    synthetic: Mapped[bool] = mapped_column(Boolean, index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (Index("events_camera_time", "camera_id", "timestamp"),)


class ExposureRow(Base):
    __tablename__ = "exposures"
    exposure_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(100), index=True)
    location_id: Mapped[str] = mapped_column(String(100))
    start: Mapped[object] = mapped_column(DateTime(timezone=True))
    end: Mapped[object] = mapped_column(DateTime(timezone=True))
    road_users: Mapped[int] = mapped_column(Integer)
    synthetic: Mapped[bool] = mapped_column(Boolean)
    payload: Mapped[dict] = mapped_column(JSON)


def connect(url: str):
    options = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, pool_pre_ping=True, connect_args=options)
    return engine, sessionmaker(engine, expire_on_commit=False)
