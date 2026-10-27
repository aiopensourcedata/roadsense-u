# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate completed ACCIDENT pilot runs against a frozen published-label manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from roadsense.evaluation.scene_benchmark import run_scene_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="Original published metadata-real.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if hashlib.sha256(args.metadata.read_bytes()).hexdigest() != plan["metadata_sha256"]:
        parser.error("Published annotation CSV differs from frozen metadata checksum")
    metadata = pd.read_csv(args.metadata)
    metadata["filename"] = metadata.path.map(lambda value: Path(value).name)
    rows, cohort, exclusions = [], [], []
    for index, annotation in enumerate(plan["selected"]):
        published = metadata[metadata.filename == annotation["filename"]]
        if (
            len(published) != 1
            or any(
                float(published.iloc[0][field]) != float(annotation[field])
                for field in ("accident_time", "accident_frame", "no_frames", "width", "height")
            )
            or published.iloc[0]["type"] != annotation["type"]
        ):
            raise ValueError("Selected label differs from original published annotation")
        path = args.runs / f"clip-{index:02d}"
        status = json.loads((path / "status.json").read_text())
        if status["status"] == "excluded_invalid_label_binding":
            exclusions.append({"file": annotation["filename"], "reasons": status["reasons"]})
            continue
        if status["status"] != "completed":
            raise ValueError("Incomplete inference; do not evaluate a partial cohort")
        observations = json.loads((path / "snapshots.json").read_text())
        if any(
            r["clip_id"] != annotation["filename"]
            or r["partition"] != annotation["partition"]
            or r["source_id"] != annotation["source_id"]
            or r["published_accident_time_s"] != annotation["accident_time"]
            for r in observations
        ):
            raise ValueError("Snapshot source/partition does not match frozen manifest")
        rows.extend(observations)
        cohort.append(annotation)
    frame = pd.DataFrame([row for row in rows if row["eligible"]])
    report = run_scene_benchmark(frame, cohort, args.output, plan["seed"])
    (args.output / "selection-audit.json").write_text(
        json.dumps(
            {
                "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                "selected_clips": len(plan["selected"]),
                "valid_binding_clips": len(cohort),
                "binding_exclusions": exclusions,
                "all_scheduled_snapshots": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "target": report["target"],
                "partitions": report["partitions"],
                "baselines": report["baselines"],
                "unavailable_baselines": report["unavailable_baselines"],
            },
            indent=2,
            allow_nan=False,
        )
    )
