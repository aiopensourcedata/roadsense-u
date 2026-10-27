# SPDX-License-Identifier: AGPL-3.0-only
"""Assess future observation coverage from a pipeline's complete frame timestamp CSV."""

import argparse
import csv
import json
from pathlib import Path

from roadsense.cloud.schemas import TrafficEvent
from roadsense.evaluation.coverage import ObservationCoverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timestamps", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-s", type=float, default=3)
    parser.add_argument("--max-gap-s", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    coverage = ObservationCoverage(args.max_gap_s)
    with args.timestamps.open() as stream:
        reader = csv.DictReader(stream)
        if not {"timestamp_s", "discontinuity"} <= set(reader.fieldnames or []):
            parser.error("Use frame_timestamps.csv, not a detection-only trajectory CSV")
        for row in reader:
            flag = row["discontinuity"].lower()
            if flag not in {"true", "false"}:
                parser.error("Discontinuity must be explicitly True/False")
            coverage.observe(float(row["timestamp_s"]), flag == "true")
    results = []
    with args.events.open() as stream:
        for line in stream:
            event = TrafficEvent.model_validate_json(line)
            result = coverage.assess(event.video_time_s, args.horizon_s)
            result["event_id"] = str(event.event_id)
            results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for result in results:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"events": len(results), "censored": sum(not x["future_observation_complete"] for x in results)}
        )
    )


if __name__ == "__main__":
    main()
