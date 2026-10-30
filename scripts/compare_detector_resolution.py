# SPDX-License-Identifier: AGPL-3.0-only
"""Inference-size/checkpoint development diagnostic on designated training sources.

No ground-truth regions or event labels enter detection, tracking or risk scoring.
Snapshot coverage is not detection accuracy or correct-participant warning recall.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from roadsense.config import PipelineConfig
from roadsense.evaluation.collision_pilot import collision_snapshots
from roadsense.evaluation.perception import perception_summary
from roadsense.pipeline import run


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--audits", required=True, help="Existing clip-NN/source-audit.json root")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 960])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--weights", type=Path, help="Optional local candidate checkpoint, without changing defaults"
    )
    parser.add_argument("--weights-sha256", help="Required exact digest for an explicit candidate checkpoint")
    args = parser.parse_args()
    plan_path, downloads_path = Path(args.plan), Path(args.downloads)
    plan = json.loads(plan_path.read_text())
    downloads = {r["filename"]: r for r in json.loads(downloads_path.read_text())}
    base = PipelineConfig.load(args.config)
    if base.synthetic or base.detector != "yolo" or base.start_frame or base.save_video:
        raise ValueError("Requires real YOLO, source origin, and no recording")
    if sha256(Path(args.config)) != plan["risk_config_sha256"]:
        raise ValueError("Base configuration differs from the selected-source experiment")
    if sha256(Path(base.weights)) != plan["weights_sha256"]:
        raise ValueError("Weights differ from selected-source experiment")
    if (args.weights is None) != (args.weights_sha256 is None):
        raise ValueError("Candidate weights and their SHA256 must be supplied together")
    if args.weights is not None:
        if sha256(args.weights) != args.weights_sha256:
            raise ValueError("Candidate checkpoint checksum mismatch")
        base.weights = str(args.weights)
    if len(set(args.indices)) != len(args.indices) or len(set(args.sizes)) != len(args.sizes):
        raise ValueError("Duplicate sources or sizes")
    if not args.indices or any(i < 0 or i >= len(plan["selected"]) for i in args.indices):
        raise ValueError("Source indices must lie within the selected cohort")
    selected = [plan["selected"][i] for i in args.indices]
    if any(row["partition"] != "train" for row in selected):
        raise ValueError("Development comparisons must use training sources only")
    for size in args.sizes:
        PipelineConfig(inference_size=size)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "development_only_not_generalization_or_actor_accuracy",
        "plan_sha256": sha256(plan_path),
        "downloads_sha256": sha256(downloads_path),
        "base_config": base.model_dump(),
        "weights_sha256": sha256(Path(base.weights)),
        "indices": args.indices,
        "sizes": args.sizes,
        "inference_end_s": 7.25,
        "snapshot_schedule_s": [1, 4, 7],
        "selected": selected,
        "order": "one worker; alternate size order between clips",
    }
    if args.weights is not None:
        manifest["reference_weights_sha256"] = plan["weights_sha256"]
        manifest["implementation_sha256"] = {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                Path("src/roadsense/config.py"),
                Path("src/roadsense/pipeline.py"),
                Path("src/roadsense/detection/detector.py"),
            ]
        }
    frozen = output / "development-plan.json"
    if frozen.exists():
        old = json.loads(frozen.read_text())
        old.pop("created_utc")
        old["base_config"] = PipelineConfig.model_validate(old["base_config"]).model_dump()
        if old != manifest:
            raise ValueError("Output already belongs to a different development plan")
    else:
        frozen.write_text(
            json.dumps({**manifest, "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2)
        )
    results = []
    for offset, (index, row) in enumerate(zip(args.indices, selected)):
        source = Path(downloads[row["filename"]]["path"])
        audit_path = Path(args.audits) / f"clip-{index:02d}" / "source-audit.json"
        audit = json.loads(audit_path.read_text())
        if not audit["valid_binding"] or sha256(source) != audit["source_sha256"]:
            raise ValueError("Invalid source binding or changed media")
        sizes = args.sizes if offset % 2 == 0 else list(reversed(args.sizes))
        for size in sizes:
            cfg = PipelineConfig.model_validate(
                {
                    **base.model_dump(),
                    "inference_size": size,
                    "max_frames": sum(t <= 7.25 for t in audit["timestamps_s"]),
                    "camera_id": "unverified-" + row["source_id"],
                    "location_id": "unknown",
                }
            )
            destination = output / f"clip-{index:02d}" / str(size)
            report_path = destination / "report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if PipelineConfig.model_validate(report["config"]).model_dump() != cfg.model_dump():
                    raise ValueError("Cached run configuration changed")
            else:
                report = run(str(source), str(destination), cfg)
            snapshots = collision_snapshots(destination, row, audit["timestamps_s"])
            (destination / "snapshots.json").write_text(json.dumps(snapshots, indent=2))
            eligible = [s for s in snapshots if s["eligible"]]
            results.append(
                {
                    "clip_id": row["filename"],
                    "index": index,
                    "inference_size": size,
                    "frames": report["frames"],
                    "elapsed_s": report["elapsed_s"],
                    "events": report["events"],
                    "eligible_snapshots": len(eligible),
                    "scoreable_snapshots": sum(s["scoreable"] for s in eligible),
                    "perception": perception_summary(destination),
                    "snapshots": snapshots,
                }
            )
            (output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps({k: v for k, v in results[-1].items() if k != "snapshots"}), flush=True)
    totals = {}
    for size in args.sizes:
        runs = [r for r in results if r["inference_size"] == size]
        snapshots = [s for r in runs for s in r["snapshots"] if s["eligible"]]
        totals[str(size)] = {
            "clips": len(runs),
            "frames": sum(r["frames"] for r in runs),
            "eligible_snapshots": len(snapshots),
            "scoreable_snapshots": sum(s["scoreable"] for s in snapshots),
            "snapshots_without_any_tracked_pair": sum(s["reported_pairs"] == 0 for s in snapshots),
            "snapshots_with_pairs_but_all_abstained": sum(
                s["reported_pairs"] > 0 and not s["scoreable"] for s in snapshots
            ),
            "events": sum(r["events"] for r in runs),
            "elapsed_s": sum(r["elapsed_s"] for r in runs),
        }
    # Exact source frame timing must match between sizes; otherwise this is not paired.
    for index in args.indices:
        rows = [r for r in results if r["index"] == index]
        timelines = [[s["prediction_time_s"] for s in r["snapshots"]] for r in rows]
        if any(t != timelines[0] for t in timelines[1:]):
            raise ValueError("Resolution runs did not preserve identical prediction timestamps")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "purpose": manifest["purpose"],
                "totals": totals,
                "paired_snapshot_timestamps_identical": True,
                "limitations": [
                    "Inspected training sources only; no independent camera evaluation.",
                    "Coverage and track counts are not actor accuracy or correct-pair warning recall.",
                    "Runtime is descriptive; initialization, export and background workloads are included.",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
