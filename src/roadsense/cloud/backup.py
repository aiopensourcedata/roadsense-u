# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded logical metadata snapshots; restore only to an empty target database."""

import json
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, func, inspect, select

from roadsense.cloud.database import AuditRow, EventRow, ExposureRow, RunRow
from roadsense.cloud.migrations import CURRENT_REVISION, migrate, state
from roadsense.provenance import canonical_digest

TABLES = [RunRow.__table__, EventRow.__table__, ExposureRow.__table__, AuditRow.__table__]
MAX_BYTES = 100 * 1024 * 1024


def backup(engine, destination: Path, max_rows: int = 100000) -> dict:
    """Snapshot one repeatable transaction, without credentials or executable data."""
    if destination.exists():
        raise ValueError("Backup destination already exists")
    tables = {}
    options = {"isolation_level": "REPEATABLE READ"} if engine.dialect.name == "postgresql" else {}
    with engine.connect().execution_options(**options) as connection, connection.begin():
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN")
        inspector = inspect(connection)
        revision = connection.scalar(select(state.c.revision)) if inspector.has_table(state.name) else 0
        total = 0
        for table in TABLES:
            if not inspector.has_table(table.name):
                continue
            rows = []
            for row in connection.execute(select(table)).mappings():
                total += 1
                if total > max_rows:
                    raise ValueError(
                        "Snapshot row limit exceeded; use an appropriately secured native database backup"
                    )
                rows.append(
                    {
                        key: value.isoformat() if isinstance(value, datetime) else value
                        for key, value in row.items()
                    }
                )
            tables[table.name] = rows
    content = {"format_version": 1, "schema_revision": revision, "tables": tables}
    document = {"sha256": canonical_digest(content), "content": content}
    data = (json.dumps(document, allow_nan=False, separators=(",", ":")) + "\n").encode()
    if len(data) > MAX_BYTES:
        raise ValueError("Snapshot size limit exceeded")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"rows": total, "bytes": len(data), "sha256": document["sha256"], "schema_revision": revision}


def restore(engine, source: Path) -> dict:
    """Verify a trusted snapshot and transactionally restore into an empty target."""
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("Snapshot size limit exceeded")
    document = json.loads(source.read_text())
    content = document["content"]
    if canonical_digest(content) != document["sha256"] or content["format_version"] != 1:
        raise ValueError("Invalid backup checksum/format")
    if not 0 <= content["schema_revision"] <= CURRENT_REVISION:
        raise ValueError("Backup schema is newer than this application")
    known = {table.name: table for table in TABLES}
    if not set(content["tables"]) <= set(known):
        raise ValueError("Unknown backup tables")
    migrate(engine)
    count = 0
    with engine.begin() as connection:
        if any(connection.scalar(select(func.count()).select_from(table)) for table in TABLES):
            raise ValueError("Restore requires an empty target; existing data will not be overwritten")
        for name, rows in content["tables"].items():
            table = known[name]
            for row in rows:
                if set(row) != set(table.columns.keys()):
                    raise ValueError("Backup row columns disagree with schema")
                for column in table.columns:
                    if isinstance(column.type, DateTime) and row[column.name] is not None:
                        row[column.name] = datetime.fromisoformat(row[column.name])
            if rows:
                connection.execute(table.insert(), rows)
            count += len(rows)
    return {"restored_rows": count, "backup_sha256": document["sha256"], "schema_revision": CURRENT_REVISION}
