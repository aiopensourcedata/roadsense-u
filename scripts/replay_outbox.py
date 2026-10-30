# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit durable-outbox recovery; successful UUIDs are acknowledged locally."""

import argparse
import sqlite3
from pathlib import Path

from roadsense.cloud.client import EventClient
from roadsense.cloud.schemas import TrafficEvent
from roadsense.config import Settings

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outbox", type=Path)
    args = p.parse_args()
    if not args.outbox.is_file():
        raise SystemExit("Outbox does not exist")
    cfg = Settings()
    client = EventClient(cfg.api_url, cfg.api_key, ca_file=cfg.api_ca_file)
    try:
        with sqlite3.connect(args.outbox) as db:
            while True:
                row = db.execute("SELECT id,body FROM pending ORDER BY rowid LIMIT 1").fetchone()
                if row is None:
                    break
                client.submit(TrafficEvent.model_validate_json(row[1]))
                db.execute("DELETE FROM pending WHERE id=?", (row[0],))
                db.commit()
        print({"submitted": client.events_sent, "body_bytes_attempted": client.bytes_attempted})
    finally:
        client.close()
