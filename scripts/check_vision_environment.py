# SPDX-License-Identifier: AGPL-3.0-only
"""Check the pinned vision boundary or write pip constraints from its JSON record."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=Path("configs/vision_environment.json"))
    parser.add_argument("--constraints", type=Path)
    args = parser.parse_args()
    versions = json.loads(args.record.read_text())["packages"]
    if args.constraints:
        args.constraints.write_text("".join(f"{name}=={version}\n" for name, version in versions.items()))
    else:
        record = json.loads(args.record.read_text())
        mismatches = (
            [] if record["python"] == f"{sys.version_info.major}.{sys.version_info.minor}" else ["python"]
        )
        for name, version in versions.items():
            try:
                current = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                current = "missing"
            if current != version:
                mismatches.append(name)
        if mismatches:
            raise SystemExit("Environment mismatch: " + ", ".join(mismatches))
        print(f"{len(versions)} pinned dependency versions match")
