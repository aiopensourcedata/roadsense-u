# SPDX-License-Identifier: AGPL-3.0-only
"""Descriptive perception coverage; requires actor annotations for accuracy claims."""

import csv
import json
from collections import Counter
from pathlib import Path


def perception_summary(run: Path) -> dict:
    """Count tracked observations and abstentions without inventing actor ground truth.

    Track-ID counts measure tracker output, not unique physical road users, ID
    switches, recall or traffic exposure. Pair observations are correlated.
    """
    report = json.loads((run / "report.json").read_text())
    classes, flags, reasons = Counter(), Counter(), Counter()
    ids, tracked_frames = set(), set()
    observations = confident = quality_pass = 0
    gate = report["config"]["min_quality"]
    with (run / "trajectories.csv").open() as stream:
        for row in csv.DictReader(stream):
            observations += 1
            classes[row["class"]] += 1
            ids.add((int(row["tracking_epoch"]), int(row["track_id"])))
            tracked_frames.add(int(row["frame"]))
            confident += float(row["confidence"]) >= gate
            quality_pass += float(row["quality"]) >= gate
            flags.update(x for x in row["flags"].split("|") if x)
    pairs = scoreable = 0
    with (run / "features.jsonl").open() as stream:
        for line in stream:
            pair = json.loads(line)
            pairs += 1
            scoreable += not pair["abstained"]
            if pair["abstained"]:
                reasons.update(pair["reasons"])
    return {
        "frames": report["frames"],
        "frames_with_tracks": len(tracked_frames),
        "tracked_observations": observations,
        "track_ids_including_epochs": len(ids),
        "observations_confidence_at_least_quality_gate": confident,
        "observations_quality_at_least_gate": quality_pass,
        "quality_gate": gate,
        "class_observations": dict(classes),
        "track_flags": dict(flags),
        "pair_observations": pairs,
        "scoreable_pair_observations": scoreable,
        "abstention_reason_counts": dict(reasons),
        "actor_accuracy": None,
        "limitations": "No actor truth; track counts are not traffic volume; correlated observations.",
    }
