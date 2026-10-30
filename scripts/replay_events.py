# SPDX-License-Identifier: AGPL-3.0-only
"""Replay persisted metadata after a connection failure, retaining original UUIDs."""

import argparse
from pathlib import Path

from roadsense.cloud.client import EventClient
from roadsense.cloud.schemas import TrafficEvent
from roadsense.config import Settings

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("events_jsonl")
    a = p.parse_args()
    cfg = Settings()
    client = EventClient(cfg.api_url, cfg.api_key, ca_file=cfg.api_ca_file)
    try:
        for line in Path(a.events_jsonl).read_text().splitlines():
            client.submit(TrafficEvent.model_validate_json(line))
        print({"submitted": client.events_sent, "body_bytes_attempted": client.bytes_attempted})
    finally:
        client.close()
