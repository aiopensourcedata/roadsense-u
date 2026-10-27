# SPDX-License-Identifier: AGPL-3.0-only
"""Build a blind actor-review bundle and optional detector-only snapshot diagnostic."""

import argparse
import csv
import importlib.metadata
import json
from bisect import bisect_left
from pathlib import Path

import cv2

from roadsense.camera.video_reader import VideoReader
from roadsense.config import PipelineConfig
from roadsense.detection.detector import YoloProposals
from roadsense.evaluation.actor_review import file_sha256
from roadsense.evaluation.actor_review_page import render_actor_page


def normalized_box(box: list | tuple, width: int, height: int) -> list[float]:
    return [max(0.0, min(1.0, float(v) / extent)) for v, extent in zip(box, [width, height, width, height])]


def prepare(
    development: Path,
    downloads_path: Path,
    audits: Path,
    output: Path,
    write_frames: bool = False,
    raw_detections: bool = False,
) -> dict:
    """Use all eligible snapshots, plus causal context, on existing training clips only."""
    if output.exists():
        raise ValueError("Choose a fresh output directory; review evidence must not be overwritten")
    plan_path = development / "development-plan.json"
    plan = json.loads(plan_path.read_text())
    summary = json.loads((development / "summary.json").read_text())
    if summary["status"] != "complete" or not summary["paired_snapshot_timestamps_identical"]:
        raise ValueError("Development comparison must be complete and paired")
    if not plan["selected"] or any(r["partition"] != "train" for r in plan["selected"]):
        raise ValueError("Actor development review requires training sources only")
    if file_sha256(downloads_path) != plan["downloads_sha256"]:
        raise ValueError("Download manifest changed")
    downloads = {r["filename"]: r for r in json.loads(downloads_path.read_text())}
    cfg = PipelineConfig.model_validate(plan["base_config"])
    if cfg.synthetic or cfg.detector != "yolo" or file_sha256(Path(cfg.weights)) != plan["weights_sha256"]:
        raise ValueError("Requires original real detector weights")
    frames, sources, snapshots, references = {}, {}, [], {}
    for index, row in zip(plan["indices"], plan["selected"], strict=True):
        source = Path(downloads[row["filename"]]["path"])
        audit_path = audits / f"clip-{index:02d}" / "source-audit.json"
        audit = json.loads(audit_path.read_text())
        if not audit["valid_binding"] or file_sha256(source) != audit["source_sha256"]:
            raise ValueError("Source audit/bytes mismatch")
        sources[index] = source
        references[str(source)] = audit["source_sha256"]
        reference_run = development / f"clip-{index:02d}" / str(plan["sizes"][0])
        snapshot_rows = json.loads((reference_run / "snapshots.json").read_text())
        times = audit["timestamps_s"]
        for snapshot in snapshot_rows:
            if not snapshot["eligible"]:
                continue
            q = snapshot["prediction_time_s"]
            snapshot_key = f"clip-{index:02d}-q{snapshot['nominal_time_s']:g}"
            q_index = bisect_left(times, q)
            if q_index == len(times) or abs(times[q_index] - q) > 1e-6:
                raise ValueError("Snapshot not bound to source PTS")
            snapshots.append({"snapshot_key": snapshot_key, "frame_key": f"clip-{index:02d}-f{q_index:06d}"})
            for offset in [-1.0, -0.5, 0.0]:
                frame_index = bisect_left(times, max(0, q + offset))
                if frame_index > q_index:
                    raise ValueError("Context cannot use future frames")
                key = f"clip-{index:02d}-f{frame_index:06d}"
                frames[key] = {
                    "frame_key": key,
                    "clip_index": index,
                    "frame_index": frame_index,
                    "timestamp_s": times[frame_index],
                    "source_sha256": audit["source_sha256"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "image": f"frames/{key}.jpg" if write_frames else None,
                }
    if not frames:
        raise ValueError("No eligible frames")
    manifest = {
        "schema_version": 1,
        "purpose": "inspected_training_actor_review_not_test_validation",
        "frames": sorted(frames.values(), key=lambda f: (f["clip_index"], f["frame_index"])),
        "raw_review_images_explicitly_enabled": write_frames,
    }
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    digest = file_sha256(manifest_path)
    template = {
        "schema_version": 1,
        "manifest_sha256": digest,
        "review_stage": "first_pass",
        "reviewer_ids": [],
        "adjudicator_id": None,
        "review_method": "human_visual",
        "frames": [
            {"frame_key": f["frame_key"], "reviewed": False, "actors": []} for f in manifest["frames"]
        ],
    }
    (output / "review-template.json").write_text(json.dumps(template, indent=2) + "\n")
    (output / "index.html").write_text(render_actor_page(manifest, digest))
    stages = {}
    if write_frames:
        (output / "frames").mkdir()
    # These are prediction data, kept separate from the score-blind page/template.
    for size in plan["sizes"]:
        stage = {key: [] for key in frames}
        for index in sources:
            run = development / f"clip-{index:02d}" / str(size)
            report = json.loads((run / "report.json").read_text())
            actual = PipelineConfig.model_validate(report["config"])
            if (
                actual.synthetic
                or actual.inference_size != size
                or report["detector_injected_for_test"]
                or any(
                    getattr(actual, field) != getattr(cfg, field)
                    for field in ("weights", "confidence", "device", "detector", "start_frame")
                )
            ):
                raise ValueError("Invalid inference provenance")
            timeline_path = run / "frame_timestamps.csv"
            with timeline_path.open() as stream:
                timeline = {int(r["frame"]): float(r["timestamp_s"]) for r in csv.DictReader(stream)}
            for f in frames.values():
                if (
                    f["clip_index"] == index
                    and abs(timeline.get(f["frame_index"], -1) - f["timestamp_s"]) > 1e-6
                ):
                    raise ValueError("Tracker timeline differs from source audit")
            tracks_path = run / "trajectories.csv"
            references[str(tracks_path)] = file_sha256(tracks_path)
            with tracks_path.open() as stream:
                for track in csv.DictReader(stream):
                    key = f"clip-{index:02d}-f{int(track['frame']):06d}"
                    if key in frames:
                        f = frames[key]
                        stage[key].append(
                            {
                                "class_name": track["class"],
                                "confidence": float(track["confidence"]),
                                "bbox": normalized_box(
                                    [track[k] for k in ["x1", "y1", "x2", "y2"]], f["width"], f["height"]
                                ),
                                "track_id": int(track["track_id"]),
                                "tracking_epoch": int(track["tracking_epoch"]),
                            }
                        )
        stages[f"tracked_{size}"] = stage
    detectors = (
        {size: YoloProposals(cfg.weights, cfg.confidence, cfg.device, size) for size in plan["sizes"]}
        if raw_detections
        else {}
    )
    for size in detectors:
        stages[f"raw_{size}"] = {key: [] for key in frames}
    if write_frames or detectors:
        for index, source in sources.items():
            wanted = {f["frame_index"]: f for f in frames.values() if f["clip_index"] == index}
            found = set()
            reader = VideoReader(str(source))
            try:
                for frame in reader:
                    if frame.index not in wanted:
                        continue
                    f = wanted[frame.index]
                    if abs(frame.timestamp - f["timestamp_s"]) > 1e-6 or frame.image.shape[:2] != (
                        f["height"],
                        f["width"],
                    ):
                        raise ValueError("Decoded review frame differs from source binding")
                    found.add(frame.index)
                    if write_frames and not cv2.imwrite(str(output / f["image"]), frame.image):
                        raise RuntimeError("Review frame export failed")
                    for size, detector in detectors.items():
                        stages[f"raw_{size}"][f["frame_key"]] = [
                            {
                                "class_name": d.class_name,
                                "bbox": normalized_box(d.bbox, f["width"], f["height"]),
                                "confidence": d.confidence,
                                "track_id": None,
                            }
                            for d in detector.detect(frame.image, frame.timestamp)
                        ]
                    if len(found) == len(wanted):
                        break
            finally:
                reader.cap.release()
            if found != set(wanted):
                raise RuntimeError("Source ended before every review frame was decoded")
    (output / "predictions.json").write_text(
        json.dumps({"manifest_sha256": digest, "stages": stages}, indent=2)
    )
    diagnostic = {}
    for size in plan["sizes"]:
        raw = stages.get(f"raw_{size}")
        tracked = stages[f"tracked_{size}"]
        diagnostic[str(size)] = {
            "snapshot_count": len(snapshots),
            "snapshots_with_two_or_more_tracked_boxes": sum(
                len(tracked[s["frame_key"]]) >= 2 for s in snapshots
            ),
            "snapshots_with_two_or_more_raw_boxes": sum(len(raw[s["frame_key"]]) >= 2 for s in snapshots)
            if raw is not None
            else None,
            "raw_pair_but_no_tracked_pair": [
                s["snapshot_key"]
                for s in snapshots
                if raw is not None and len(raw[s["frame_key"]]) >= 2 and len(tracked[s["frame_key"]]) < 2
            ],
        }
    result = {
        "status": "complete",
        "manifest_sha256": digest,
        "n_review_frames": len(frames),
        "n_snapshots": len(snapshots),
        "diagnostic": diagnostic,
        "actor_accuracy": None,
        "raw_frames_retained": write_frames,
        "limitation": "Model box counts are not physical actor truth or detector accuracy.",
    }
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2))
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "development_plan_sha256": file_sha256(plan_path),
                "weights_sha256": plan["weights_sha256"],
                "references_sha256": references,
                "snapshot_frames": snapshots,
                "raw_detection_config": {
                    "confidence": cfg.confidence,
                    "sizes": plan["sizes"],
                    "device": cfg.device,
                },
                "runtime_versions": {
                    package: importlib.metadata.version(package)
                    for package in [
                        "ultralytics-opencv-headless",
                        "torch",
                        "opencv-python-headless",
                        "numpy",
                    ]
                }
                if raw_detections
                else {},
                "implementation_sha256": {
                    str(path): file_sha256(path)
                    for path in [
                        Path(__file__),
                        Path("src/roadsense/detection/detector.py"),
                    ]
                },
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--write-review-frames", action="store_true", help="Explicitly retain sampled raw JPEGs locally"
    )
    parser.add_argument(
        "--raw-detections", action="store_true", help="Run independent YOLO.predict diagnostics"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.development,
                args.downloads,
                args.audits,
                args.output,
                args.write_review_frames,
                args.raw_detections,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
