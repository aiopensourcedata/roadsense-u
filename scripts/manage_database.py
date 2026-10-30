# SPDX-License-Identifier: AGPL-3.0-only
"""Local schema and metadata recovery commands; credentials come from environment."""

import argparse
import json
from pathlib import Path

from roadsense.cloud.backup import backup, restore
from roadsense.cloud.database import connect
from roadsense.cloud.migrations import migrate, require_current
from roadsense.config import Settings

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["upgrade", "revision", "downgrade", "backup", "restore"])
    p.add_argument("--path", type=Path)
    p.add_argument("--target", type=int)
    p.add_argument("--allow-data-loss", action="store_true")
    args = p.parse_args()
    engine, _ = connect(Settings().database_url)
    try:
        if args.action == "upgrade":
            result = {"revision": migrate(engine)}
        elif args.action == "revision":
            result = {"revision": require_current(engine)}
        elif args.action == "downgrade":
            if args.target is None:
                p.error("--target required")
            result = {"revision": migrate(engine, args.target, allow_data_loss=args.allow_data_loss)}
        else:
            if args.path is None:
                p.error("--path required")
            result = backup(engine, args.path) if args.action == "backup" else restore(engine, args.path)
        print(json.dumps(result, indent=2))
    finally:
        engine.dispose()
