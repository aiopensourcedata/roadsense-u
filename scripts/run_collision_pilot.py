# SPDX-License-Identifier: AGPL-3.0-only
"""Process a frozen, locally downloaded ACCIDENT sample without uploading footage."""

import argparse
import hashlib
import json
from pathlib import Path

from roadsense.config import PipelineConfig
from roadsense.evaluation.collision_pilot import audit_collision_source, collision_snapshots
from roadsense.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--inference-end-s",
        type=float,
        default=11.0,
        help="Process through at least the final scheduled snapshot; source audit always covers the full video",
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["snapshot_schedule_nominal_s"] != [1.0, 4.0, 7.0] or plan["false_positive_budget"] != 0.05:
        parser.error("This runner implements only the frozen 1/4/7-second, 5% validation-budget pilot")
    if not 7.25 <= args.inference_end_s <= 11:
        parser.error("Frozen pilot permits inference ends from 7.25 through 11 seconds only")
    downloads = {row["filename"]: row for row in json.loads(args.downloads.read_text())}
    cfg = PipelineConfig.load(args.config)
    if cfg.horizon_s != 3 or cfg.alert_threshold != 0.45:
        parser.error("This pilot fixes the three-second horizon and 0.45 regression threshold")
    if hashlib.sha256(args.config.read_bytes()).hexdigest() != plan["risk_config_sha256"]:
        parser.error("Runtime configuration differs from frozen plan")
    if hashlib.sha256(Path(cfg.weights).read_bytes()).hexdigest() != plan["weights_sha256"]:
        parser.error("Detector weights differ from frozen plan")
    if cfg.synthetic or cfg.detector != "yolo" or cfg.save_video or cfg.start_frame:
        parser.error("Pilot requires actual YOLO, start frame zero and no recording")
    indices = args.indices if args.indices is not None else list(range(len(plan["selected"])))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(plan["selected"]) for i in indices):
        parser.error("Indices must be distinct valid sample indices")
    for index in indices:
        row = plan["selected"][index]
        download = downloads[row["filename"]]
        source = Path(download["path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != download["sha256"]:
            raise ValueError("Source checksum differs from download manifest")
        dest = args.output / f"clip-{index:02d}"
        if dest.exists() and not args.resume:
            parser.error("Clip output already exists; use --resume or a fresh output directory")
        dest.mkdir(parents=True, exist_ok=True)
        audit_path = dest / "source-audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
        else:
            audit = audit_collision_source(source, row)
            audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        if audit["source_sha256"] != download["sha256"]:
            raise ValueError("Cached source audit belongs to another file")
        if not audit["valid_binding"]:
            status = {"status": "excluded_invalid_label_binding", "reasons": audit["exclusions"]}
        else:
            config = cfg.model_copy(deep=True)
            config.camera_id = "unverified-" + row["source_id"]
            config.location_id = "unknown"
            config.max_frames = sum(t <= args.inference_end_s for t in audit["timestamps_s"])
            run_path = dest / "run"
            if (run_path / "report.json").exists():
                report = json.loads((run_path / "report.json").read_text())
                if PipelineConfig.model_validate(report["config"]).model_dump() != config.model_dump():
                    raise ValueError("Cached run has a different runtime configuration")
            else:
                report = run(str(source), str(run_path), config)
            snapshots = collision_snapshots(run_path, row, audit["timestamps_s"])
            (dest / "snapshots.json").write_text(json.dumps(snapshots, indent=2) + "\n")
            status = {
                "status": "completed",
                "frames": report["frames"],
                "elapsed_s": report["elapsed_s"],
                "fps": report["throughput_fps"],
                "source_file": row["filename"],
                "partition": row["partition"],
                "snapshots": len(snapshots),
                "eligible_snapshots": sum(x["eligible"] for x in snapshots),
                "scoreable_eligible_snapshots": sum(x["eligible"] and x["scoreable"] for x in snapshots),
            }
        (dest / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps({"index": index, **status}), flush=True)


if __name__ == "__main__":
    main()
