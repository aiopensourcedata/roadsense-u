# SPDX-License-Identifier: AGPL-3.0-only
"""Run a controlled, explicitly synthetic diagnosis of trajectory-quality coverage."""

import argparse
import json
from pathlib import Path

from roadsense.evaluation.motion_diagnostic import diagnose_motion_quality

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/motion-quality-diagnostic.json"))
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists; choose a new path")
    result = diagnose_motion_quality(a.repeats, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                x["scenario"]: {k: round(v["value"], 4) for k, v in x["metrics"].items()}
                for x in result["scenarios"]
            },
            indent=2,
        )
    )
