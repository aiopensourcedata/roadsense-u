# SPDX-License-Identifier: AGPL-3.0-only
"""Ablate the identity guard on cached boxes/features without rerunning inference.

This is a feature replay, not an end-to-end FPS benchmark or accuracy evaluation.
It preserves the reference geometric/quality scores and applies only the new guard.
"""

import argparse
import csv
import json
import time
from itertools import combinations, groupby
from pathlib import Path

from roadsense.config import PipelineConfig
from roadsense.detection.pair_identity import PairIdentityGuard
from roadsense.detection.tracker import Detection
from roadsense.evaluation.coverage import ObservationCoverage


def compare(reference: Path, output: Path, timestamps: Path | None = None) -> dict:
    report = json.loads((reference / "report.json").read_text())
    cfg = PipelineConfig(**report["config"])
    if report["config"].get("pair_identity", {}).get("enabled", False):
        raise ValueError("Reference must precede identity gating or explicitly disable it")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    cfg.pair_identity = cfg.pair_identity.model_copy(update={"enabled": True})
    guard = PairIdentityGuard(cfg.pair_identity)
    coverage = None
    if timestamps is not None:
        coverage = ObservationCoverage(cfg.observation_max_gap_s)
        with timestamps.open() as stream:
            reader = csv.DictReader(stream)
            if not {"timestamp_s", "discontinuity"} <= set(reader.fieldnames or []):
                raise ValueError("Coverage needs complete frame timestamps, not detections")
            for row in reader:
                flag = row["discontinuity"].lower()
                if flag not in {"true", "false"}:
                    raise ValueError("Discontinuity requires explicit True/False")
                coverage.observe(float(row["timestamp_s"]), flag == "true")
    output.mkdir(parents=True)
    last = {"baseline": {}, "guarded": {}}
    alerts = {"baseline": [], "guarded": []}
    counts = {
        "pair_observations": 0,
        "baseline_accepted": 0,
        "guarded_accepted": 0,
        "identity_ambiguous": 0,
        "new_abstentions_from_previously_accepted": 0,
    }
    guard_seconds = 0.0
    previous_epoch = None
    with (
        (reference / "trajectories.csv").open() as trajectories,
        (reference / "features.jsonl").open() as features,
        (output / "ambiguities.jsonl").open("w") as ambiguous_file,
    ):
        rows = csv.DictReader(trajectories)
        feature_iter = (json.loads(line) for line in features)
        for _, group in groupby(rows, key=lambda r: (int(r["tracking_epoch"]), int(r["frame"]))):
            group = list(group)
            epoch = int(group[0]["tracking_epoch"])
            timestamp = float(group[0]["timestamp_s"])
            if epoch != previous_epoch:
                guard.clear()
                last = {"baseline": {}, "guarded": {}}
                previous_epoch = epoch
            tick = time.perf_counter()
            guard.begin_frame(timestamp)
            guard_seconds += time.perf_counter() - tick
            for a, b in combinations(group, 2):
                if a["units"] != b["units"]:
                    continue
                old = next(feature_iter)
                ids = [int(a["track_id"]), int(b["track_id"])]
                if old["track_ids"] != ids or abs(old["timestamp_s"] - timestamp) > 1e-8:
                    raise ValueError("Cached trajectories/features do not match in order or timestamp")
                if int(old["tracking_epoch"]) != epoch:
                    raise ValueError("Cached features have a different tracking epoch")

                def detection(row):
                    return Detection(
                        tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")),
                        row["class"],
                        float(row["confidence"]),
                        int(row["track_id"]),
                    )

                tick = time.perf_counter()
                assessment = guard.assess(detection(a), detection(b), timestamp)
                guard_seconds += time.perf_counter() - tick
                counts["pair_observations"] += 1
                counts["baseline_accepted"] += not old["abstained"]
                counts["identity_ambiguous"] += assessment.ambiguous
                counts["guarded_accepted"] += not old["abstained"] and not assessment.ambiguous
                counts["new_abstentions_from_previously_accepted"] += (
                    not old["abstained"] and assessment.ambiguous
                )
                if assessment.ambiguous:
                    ambiguous_file.write(
                        json.dumps(
                            {
                                "timestamp_s": timestamp,
                                "tracking_epoch": epoch,
                                "track_ids": ids,
                                "previously_accepted": not old["abstained"],
                                "previous_score": old["risk_score"],
                                "measurements": assessment.measurements,
                            }
                        )
                        + "\n"
                    )
                key = tuple(sorted(ids))
                for mode in ("baseline", "guarded"):
                    if old["abstained"] or (mode == "guarded" and assessment.ambiguous):
                        continue
                    if (
                        old["risk_score"] < cfg.alert_threshold
                        or timestamp - last[mode].get(key, -1e9) < cfg.cooldown_s
                    ):
                        continue
                    item = {
                        "timestamp_s": timestamp,
                        "tracking_epoch": epoch,
                        "track_ids": ids,
                        "heuristic_score": old["risk_score"],
                    }
                    if coverage is not None:
                        item["coverage"] = coverage.assess(timestamp, cfg.horizon_s)
                    alerts[mode].append(item)
                    last[mode][key] = timestamp
            for mode in last:
                last[mode] = {
                    key: value for key, value in last[mode].items() if timestamp - value < 2 * cfg.cooldown_s
                }
        if next(feature_iter, None) is not None:
            raise ValueError("Unmatched cached feature rows")
    reference_events = [
        json.loads(x) for x in (reference / "events.jsonl").read_text().splitlines() if x.strip()
    ]

    def signatures(items, time_key, ids_key, score_key):
        return sorted(
            (round(x[time_key], 8), x["tracking_epoch"], tuple(sorted(x[ids_key])), round(x[score_key], 12))
            for x in items
        )

    if signatures(reference_events, "video_time_s", "track_ids", "risk_score") != signatures(
        alerts["baseline"], "timestamp_s", "track_ids", "heuristic_score"
    ):
        raise ValueError("Baseline replay did not reproduce recorded alert emissions")
    result = {
        "mode": "cached_boxes_and_features_ablation",
        "synthetic": report["synthetic"],
        "baseline_alerts_reproduced": True,
        "reference_session_id": report["session_id"],
        "risk_thresholds_unchanged": True,
        "identity_config": cfg.pair_identity.model_dump(),
        "counts": counts,
        "alerts": alerts,
        "guard_timing_includes": "Cached box reconstruction and pair checks; excludes decoding, detection and JSON replay.",
        "identity_guard_seconds": guard_seconds,
        "identity_guard_ms_per_reference_frame": 1000 * guard_seconds / report["frames"],
        "limitations": [
            "No detector/decode rerun; timing excludes them.",
            "Development regression is not held-out accuracy evidence.",
            "Fewer alerts and lower coverage are not proof of better conflict detection.",
            "Unchanged scores are copied from reference features, not independently recomputed.",
        ],
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.output, args.timestamps)
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "alerts": {k: len(v) for k, v in result["alerts"].items()},
                "identity_guard_ms_per_reference_frame": result["identity_guard_ms_per_reference_frame"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
