# SPDX-License-Identifier: AGPL-3.0-only
"""Small explicit additive schema revisions, serialized across API workers."""

from sqlalchemy import Column, Integer, MetaData, Table, inspect, select, text

from roadsense.cloud.database import AuditRow, EventRow, ExposureRow, RunRow

CURRENT_REVISION = 2
state = Table(
    "schema_revision",
    MetaData(),
    Column("singleton", Integer, primary_key=True),
    Column("revision", Integer, nullable=False),
)
REVISIONS = {1: [EventRow.__table__, ExposureRow.__table__, RunRow.__table__], 2: [AuditRow.__table__]}


def migrate(engine, target: int = CURRENT_REVISION, *, allow_data_loss: bool = False) -> int:
    if target not in range(CURRENT_REVISION + 1):
        raise ValueError("Unsupported schema revision")
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(text("SELECT pg_advisory_xact_lock(571930111)"))
        elif engine.dialect.name == "sqlite":
            connection.execute(text("BEGIN IMMEDIATE"))
        state.create(connection, checkfirst=True)
        current = connection.scalar(select(state.c.revision)) or 0
        if current > CURRENT_REVISION:
            raise ValueError("Database is newer than this application")
        if target < current and not allow_data_loss:
            raise ValueError(
                "Downgrade requires explicit allow_data_loss on a disposable/restorable database"
            )
        for revision in range(current + 1, target + 1):
            for table in REVISIONS[revision]:
                table.create(connection, checkfirst=True)
                actual = {column["name"] for column in inspect(connection).get_columns(table.name)}
                if not set(table.columns.keys()) <= actual:
                    raise ValueError(
                        f"Existing table {table.name} is incompatible; explicit migration required"
                    )
        for revision in range(current, target, -1):
            for table in reversed(REVISIONS[revision]):
                table.drop(connection, checkfirst=True)
        connection.execute(state.delete())
        connection.execute(state.insert().values(singleton=1, revision=target))
    return target


def require_current(engine) -> int:
    with engine.connect() as connection:
        if not inspect(connection).has_table(state.name):
            raise ValueError("Database is unmigrated; run scripts/migrate_database.py upgrade")
        revision = connection.scalar(select(state.c.revision))
    if revision != CURRENT_REVISION:
        raise ValueError("Database revision mismatch; migrate explicitly before starting the API")
    return revision
