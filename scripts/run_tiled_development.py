# SPDX-License-Identifier: AGPL-3.0-only
"""Frozen tiled-perception development comparison; no fresh-test or actor-accuracy claim."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from roadsense.config import PipelineConfig
from roadsense.evaluation.actor_review import file_sha256
from roadsense.evaluation.collision_pilot import collision_snapshots
from roadsense.evaluation.perception import perception_summary
from roadsense.pipeline import run


def pair_records(path: Path, end: float) -> list[dict]:
    with (path / "features.jsonl").open() as stream:
        return [
            {k: v for k, v in row.items() if k != "projection"}
            for line in stream
            if (row := json.loads(line))["timestamp_s"] <= end + 1e-9
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = args.development / "development-plan.json"
    plan = json.loads(original.read_text())
    if plan["indices"] != [0, 4, 8, 12] or any(row["partition"] != "train" for row in plan["selected"]):
        raise ValueError("This fixed development protocol requires the original four training sources")
    if file_sha256(args.downloads) != plan["downloads_sha256"]:
        raise ValueError("Download manifest changed")
    base = PipelineConfig.model_validate(plan["base_config"])
    if (
        base.synthetic
        or base.detector != "yolo"
        or base.tiling.enabled
        or base.start_frame
        or base.save_video
    ):
        raise ValueError("Requires real default YOLO development data without recordings")
    if file_sha256(Path(base.weights)) != plan["weights_sha256"]:
        raise ValueError("Weights changed")
    references = {}
    for index in plan["indices"]:
        for name in ["report.json", "features.jsonl", "frame_timestamps.csv", "snapshots.json"]:
            p = args.development / f"clip-{index:02d}" / "640" / name
            references[str(p)] = file_sha256(p)
    sources = {row["filename"]: row for row in json.loads(args.downloads.read_text())}
    manifest = {
        "purpose": "inspected_training_development_not_independent_test",
        "original_plan_sha256": file_sha256(original),
        "references_sha256": references,
        "weights_sha256": plan["weights_sha256"],
        "indices": plan["indices"],
        "base_config": base.model_dump(),
        "candidate": {"grid": 2, "overlap": 0.2, "nms_iou": 0.6},
        "control": "full-frame explicit ByteTrack bridge on clip-12; require exact prior pair records",
        "end_media_s": 7.25,
        "schedule_s": [1, 4, 7],
        "code_sha256": {
            str(p): file_sha256(p)
            for p in [
                Path(__file__),
                Path("src/roadsense/config.py"),
                Path("src/roadsense/pipeline.py"),
                Path("src/roadsense/detection/tiled.py"),
                Path("src/roadsense/detection/detector.py"),
            ]
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "frozen-plan.json"
    if frozen.exists():
        previous = json.loads(frozen.read_text())
        previous.pop("created_utc")
        if previous != manifest:
            raise ValueError("Existing output has a different frozen protocol or implementation")
    else:
        frozen.write_text(
            json.dumps({**manifest, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2)
        )
    results = []
    # Bridge control runs before the candidate; do not interpret crop effects if it fails.
    jobs = [(12, plan["selected"][3], 1, "bridge_control")]
    jobs += [(i, row, 2, "tiled") for i, row in zip(plan["indices"], plan["selected"], strict=True)]
    for index, row, grid, profile in jobs:
        source = Path(sources[row["filename"]]["path"])
        audit = json.loads((args.audits / f"clip-{index:02d}" / "source-audit.json").read_text())
        if not audit["valid_binding"] or file_sha256(source) != audit["source_sha256"]:
            raise ValueError("Invalid source binding")
        config = PipelineConfig.model_validate(
            {
                **base.model_dump(),
                "camera_id": "unverified-" + row["source_id"],
                "location_id": "unknown",
                "inference_size": 640,
                "max_frames": sum(t <= 7.25 for t in audit["timestamps_s"]),
                "tiling": {"enabled": True, "grid": grid, "overlap": 0.2, "nms_iou": 0.6},
            }
        )
        dest = args.output / profile / f"clip-{index:02d}"
        if (dest / "report.json").exists():
            report = json.loads((dest / "report.json").read_text())
            if PipelineConfig.model_validate(report["config"]) != config:
                raise ValueError("Cached config changed")
        else:
            report = run(str(source), str(dest), config)
        baseline = args.development / f"clip-{index:02d}" / "640"
        if profile == "bridge_control":
            end = report["video_duration_s"]
            reference = pair_records(baseline, end)
            actual = pair_records(dest, end)
            if reference != actual:
                raise ValueError("Tracker bridge differs from immutable baseline; do not run candidate")
            (args.output / "bridge-verification.json").write_text(
                json.dumps(
                    {
                        "identical_existing_pair_records": True,
                        "pairs_compared": len(reference),
                        "index": index,
                        "source_end_s": end,
                    },
                    indent=2,
                )
            )
        snapshots = collision_snapshots(dest, row, audit["timestamps_s"])
        original_snapshots = json.loads((baseline / "snapshots.json").read_text())
        if [s["prediction_time_s"] for s in snapshots] != [
            s["prediction_time_s"] for s in original_snapshots
        ]:
            raise ValueError("Unpaired snapshot timing")
        (dest / "snapshots.json").write_text(json.dumps(snapshots, indent=2))
        eligible = [s for s in snapshots if s["eligible"]]
        result = {
            "profile": profile,
            "index": index,
            "clip_id": row["filename"],
            "frames": report["frames"],
            "events": report["events"],
            "elapsed_s": report["elapsed_s"],
            "model_calls": report["perception_diagnostics"]["model_calls"],
            "eligible_snapshots": len(eligible),
            "scoreable_snapshots": sum(s["scoreable"] for s in eligible),
            "scoreable_positive_snapshots": sum(s["scoreable"] and s["label"] == 1 for s in eligible),
            "scoreable_negative_snapshots": sum(s["scoreable"] and s["label"] == 0 for s in eligible),
            "perception": perception_summary(dest),
            "snapshots": snapshots,
        }
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(
            json.dumps({k: v for k, v in result.items() if k not in ["perception", "snapshots"]}), flush=True
        )
    candidate = [r for r in results if r["profile"] == "tiled"]
    baseline_rows = json.loads((args.development / "results.json").read_text())
    comparisons = []
    for index in plan["indices"]:
        old = next(r for r in baseline_rows if r["index"] == index and r["inference_size"] == 640)
        new = next(r for r in candidate if r["index"] == index)
        comparisons.append(
            {
                "index": index,
                "baseline_usable": old["scoreable_snapshots"],
                "candidate_usable": new["scoreable_snapshots"],
                "eligible": new["eligible_snapshots"],
                "baseline_events": old["events"],
                "candidate_events": new["events"],
            }
        )
    summary = {
        "status": "complete",
        "purpose": manifest["purpose"],
        "actor_accuracy": None,
        "new_inferred_frames": sum(r["frames"] for r in results),
        "model_calls": sum(r["model_calls"] for r in results),
        "candidate_usable": sum(r["scoreable_snapshots"] for r in candidate),
        "eligible_snapshots": sum(r["eligible_snapshots"] for r in candidate),
        "comparisons": comparisons,
        "limitations": [
            "Development coverage and event counts are not improved actor accuracy or warning precision.",
            "No independent actor labels or fresh camera evaluation; no risk thresholds changed.",
            "Cached baseline runtime is not a controlled latency comparator.",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)


if __name__ == "__main__":
    main()
