# SPDX-License-Identifier: AGPL-3.0-only
"""Refit and reproduce a cached scene report without detector inference or media."""

import argparse
import json
from pathlib import Path

import pandas as pd

from roadsense.evaluation.scene_benchmark import run_scene_benchmark

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    expected = json.loads(args.expected.read_text())
    report = run_scene_benchmark(
        pd.read_csv(args.observations), json.loads(args.cohort.read_text()), args.output
    )
    if report != expected:
        raise SystemExit("Report differs from expected evidence; inspect outputs")
    print("Full cached report and bootstrap exactly reproduced; detector/motion inference was not rerun")
