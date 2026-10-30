# SPDX-License-Identifier: AGPL-3.0-only
"""Add tiled predictions to a completed development bundle without changing its labels/images."""

import argparse
import csv
import json
from pathlib import Path

from roadsense.camera.video_reader import VideoReader
from roadsense.config import PipelineConfig
from roadsense.detection.tiled import TiledYoloProposals
from roadsense.evaluation.actor_review import bind_prediction_frames, file_sha256


def normalize(box: tuple | list, width: int, height: int) -> list[float]:
    return [max(0.0, min(1.0, float(v) / size)) for v, size in zip(box, [width, height, width, height])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capacity-bundle", type=Path, help="Optional same-frame larger-checkpoint prediction bundle"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output file")
    manifest_path = args.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = file_sha256(manifest_path)
    prior_path = args.bundle / "predictions.json"
    data = json.loads(prior_path.read_text())
    plan_path = args.development / "frozen-plan.json"
    plan = json.loads(plan_path.read_text())
    if json.loads((args.development / "summary.json").read_text())["status"] != "complete":
        raise ValueError("Candidate development run is incomplete")
    if plan["purpose"] != "inspected_training_development_not_independent_test":
        raise ValueError("Expected an explicitly development-only candidate")
    original_roots = {Path(p).parents[2] for p in plan["references_sha256"] if Path(p).name == "report.json"}
    if len(original_roots) != 1:
        raise ValueError("Cannot resolve original development provenance")
    original_plan_path = next(iter(original_roots)) / "development-plan.json"
    if file_sha256(original_plan_path) != plan["original_plan_sha256"]:
        raise ValueError("Original cohort selection changed")
    original_plan = json.loads(original_plan_path.read_text())
    if file_sha256(args.downloads) != original_plan["downloads_sha256"]:
        raise ValueError("Source download manifest changed")
    original_rows = dict(zip(original_plan["indices"], original_plan["selected"], strict=True))
    tiled_code = Path("src/roadsense/detection/tiled.py")
    if file_sha256(tiled_code) != plan["code_sha256"][str(tiled_code)]:
        raise ValueError("Raw tiled detector differs from the candidate implementation")
    if data["manifest_sha256"] != digest:
        raise ValueError("Prior predictions have a different manifest")
    frames = {f["frame_key"]: f for f in manifest["frames"]}
    if len(frames) != len(manifest["frames"]) or any(
        set(stage) != set(frames) for stage in data["stages"].values()
    ):
        raise ValueError("Duplicate or mismatched review frames")
    # Preserve all original stages and their manifest; the blind page and labels are untouched.
    if {"raw_tiled_640", "tracked_tiled_640"} & data["stages"].keys():
        raise ValueError("Tiled stages already exist")
    source_rows = json.loads(args.downloads.read_text())
    paths = {r["sha256"]: Path(r["path"]) for r in source_rows}
    tracked, raw = {k: [] for k in frames}, {k: [] for k in frames}
    references = {str(prior_path): file_sha256(prior_path), str(plan_path): file_sha256(plan_path)}
    detector = None
    model_calls = 0
    for index in sorted({f["clip_index"] for f in frames.values()}):
        if index not in plan["indices"]:
            raise ValueError("Review source absent from candidate cohort")
        run = args.development / "tiled" / f"clip-{index:02d}"
        report = json.loads((run / "report.json").read_text())
        if report["camera_id"] != "unverified-" + original_rows[index]["source_id"]:
            raise ValueError("Candidate run belongs to a different source")
        cfg = PipelineConfig.model_validate(report["config"])
        expected = PipelineConfig.model_validate(
            {**plan["base_config"], "inference_size": 640, "tiling": {"enabled": True, **plan["candidate"]}}
        )
        for field in ("weights", "confidence", "device", "tiling", "inference_size", "detector"):
            if getattr(cfg, field) != getattr(expected, field):
                raise ValueError("Candidate configuration differs from frozen plan")
        if (
            cfg.synthetic
            or report["detector_injected_for_test"]
            or file_sha256(Path(cfg.weights)) != plan["weights_sha256"]
        ):
            raise ValueError("Invalid candidate inference provenance")
        if detector is None:
            detector = TiledYoloProposals(
                cfg.weights,
                cfg.confidence,
                cfg.device,
                cfg.inference_size,
                cfg.tiling.grid,
                cfg.tiling.overlap,
                cfg.tiling.nms_iou,
            )
        wanted = {f["frame_index"]: f for f in frames.values() if f["clip_index"] == index}
        hashes = {f["source_sha256"] for f in wanted.values()}
        if len(hashes) != 1:
            raise ValueError("One exact source hash required per clip")
        source_hash = next(iter(hashes))
        source = paths[source_hash]
        if file_sha256(source) != source_hash:
            raise ValueError("Source bytes changed")
        references[str(source)] = source_hash
        track_path = run / "trajectories.csv"
        references[str(track_path)] = file_sha256(track_path)
        for name in ("report.json", "frame_timestamps.csv"):
            references[str(run / name)] = file_sha256(run / name)
        with (run / "frame_timestamps.csv").open() as stream:
            timeline = {int(r["frame"]): float(r["timestamp_s"]) for r in csv.DictReader(stream)}
        for frame_index, f in wanted.items():
            if abs(timeline.get(frame_index, -1) - f["timestamp_s"]) > 1e-6:
                raise ValueError("Candidate/review frame timestamps differ")
        with track_path.open() as stream:
            for row in csv.DictReader(stream):
                f = wanted.get(int(row["frame"]))
                if f is not None:
                    tracked[f["frame_key"]].append(
                        {
                            "class_name": row["class"],
                            "confidence": float(row["confidence"]),
                            "track_id": int(row["track_id"]),
                            "tracking_epoch": int(row["tracking_epoch"]),
                            "bbox": normalize(
                                [row[k] for k in ["x1", "y1", "x2", "y2"]], f["width"], f["height"]
                            ),
                        }
                    )
        found = set()
        reader = VideoReader(str(source))
        try:
            for frame in reader:
                f = wanted.get(frame.index)
                if f is None:
                    continue
                if abs(frame.timestamp - f["timestamp_s"]) > 1e-6 or frame.image.shape[:2] != (
                    f["height"],
                    f["width"],
                ):
                    raise ValueError("Decoded candidate frame differs from review")
                raw[f["frame_key"]] = [
                    {
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "track_id": None,
                        "bbox": normalize(d.bbox, f["width"], f["height"]),
                    }
                    for d in detector.detect(frame.image, frame.timestamp)
                ]
                model_calls += detector.last_diagnostics["inference_views"]
                found.add(frame.index)
                if found == set(wanted):
                    break
        finally:
            reader.cap.release()
        if found != set(wanted):
            raise ValueError("Incomplete candidate frame decode")
    data["stages"].update(raw_tiled_640=raw, tracked_tiled_640=tracked)
    if args.capacity_bundle is not None:
        candidate_predictions = args.capacity_bundle / "predictions.json"
        rebound = bind_prediction_frames(
            manifest_path, args.capacity_bundle / "manifest.json", candidate_predictions
        )
        if set(rebound["stages"]) != {"raw_640", "tracked_640"}:
            raise ValueError("Capacity bundle must contain only raw/tracked 640 stages")
        data["stages"].update(
            raw_capacity_640=rebound["stages"]["raw_640"],
            tracked_capacity_640=rebound["stages"]["tracked_640"],
        )
        data["capacity_rebinding"] = rebound["rebinding_provenance"]
        references[str(candidate_predictions)] = file_sha256(candidate_predictions)
    data["extension_provenance"] = {
        "scope": "training_development_not_actor_truth",
        "references_sha256": references,
        "model_calls": model_calls,
        "raw_frames_retained": False,
        "implementation_sha256": {
            str(p): file_sha256(p) for p in [Path(__file__), Path("src/roadsense/detection/tiled.py")]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest_sha256": digest,
                "frames_per_stage": len(frames),
                "stages": list(data["stages"]),
                "model_calls": model_calls,
                "actor_accuracy": None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
