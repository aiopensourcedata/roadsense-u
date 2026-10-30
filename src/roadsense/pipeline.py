# SPDX-License-Identifier: AGPL-3.0-only
"""Video -> tracked trajectories -> measured interactions -> compact event metadata."""

import csv
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import psutil

from roadsense.calibration.ground_plane import GroundPlane
from roadsense.camera.preview import LivePreview
from roadsense.camera.sources import parse_source
from roadsense.camera.video_reader import VideoReader
from roadsense.cloud.client import EventClient
from roadsense.cloud.outbox import DeliveryOutbox
from roadsense.cloud.schemas import TrafficEvent
from roadsense.config import PipelineConfig
from roadsense.detection.detector import SyntheticDetector, YoloByteTrack
from roadsense.detection.pair_identity import PairIdentityGuard
from roadsense.detection.tiled import TiledYoloByteTrack
from roadsense.evaluation.coverage import ObservationCoverage
from roadsense.motion.features import pair_features, pair_projection
from roadsense.motion.trajectories import Trajectories
from roadsense.provenance import build_manifest
from roadsense.risk.rules import estimate

log = logging.getLogger(__name__)


def run(
    source: str,
    output: str,
    config: PipelineConfig,
    client: EventClient | None = None,
    video_start: datetime | None = None,
    detector_override=None,
    preview: LivePreview | None = None,
    realtime: bool = False,
) -> dict:
    """Process once with bounded motion history. Raw frames are never saved by default.

    File wall timestamps default to ingestion start (not capture time). Supply video_start for real capture time.
    Injection of a detector is intended for integration tests only and is recorded in the report.
    """
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "report.json").exists() or (destination / "trajectories.csv").exists():
        raise ValueError("Output directory already contains a run; choose a new directory")
    reader = None
    try:
        resolved_source = parse_source(source)
        reader = VideoReader(resolved_source, start_frame=config.start_frame)
        if video_start is not None and video_start.tzinfo is None:
            raise ValueError("video_start must have a timezone")
        detector = detector_override or (
            SyntheticDetector()
            if config.detector == "synthetic"
            else (
                TiledYoloByteTrack(
                    config.weights,
                    config.confidence,
                    config.device,
                    config.inference_size,
                    config.tiling.grid,
                    config.tiling.overlap,
                    config.tiling.nms_iou,
                )
                if config.tiling.enabled
                else YoloByteTrack(config.weights, config.confidence, config.device, config.inference_size)
            )
        )
        plane = GroundPlane.load(config.calibration) if config.calibration else None
    except BaseException as exc:
        if reader is not None:
            reader.cap.release()
        report = {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "failure_stage": "initialization",
            "error_type": type(exc).__name__,
            "config": config.model_dump(),
            "camera_id": config.camera_id,
            "location_id": config.location_id,
            "synthetic": config.synthetic,
            "frames": 0,
            "events": 0,
            "throughput_fps": 0,
            "run_manifest_sha256": None,
        }
        temporary = destination / "report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(destination / "report.json")
        (destination / "event_evaluation.jsonl").touch()
        raise
    motion, session_id = Trajectories(), uuid4()
    identity = PairIdentityGuard(config.pair_identity)
    coverage = ObservationCoverage(config.observation_max_gap_s)
    identity_abstentions = 0
    outside_hull = mixed_coordinate_pairs = 0
    manifest = None
    geometry = None
    decode_latencies, inference_latencies, arrival_ages = (
        deque(maxlen=10000),
        deque(maxlen=10000),
        deque(maxlen=10000),
    )
    scoring_latencies = deque(maxlen=10000)
    origin = video_start or datetime.now(timezone.utc)
    if origin.tzinfo is None:
        reader.cap.release()
        raise ValueError("video_start must have a timezone")
    writer = None
    frames = emitted = abstained = submission_failures = 0
    latencies, last_events = deque(maxlen=10000), {}
    tracking_epoch = track_segments = 0
    initial_network_bytes = client.bytes_attempted if client else 0
    started, cpu_started = time.perf_counter(), time.process_time()
    first_timestamp, last_timestamp = None, 0.0
    first_source_frame, last_source_frame = None, None
    fields = [
        "frame",
        "timestamp_s",
        "timestamp_source",
        "track_id",
        "tracking_epoch",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_x",
        "image_y",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "units",
        "quality",
        "residual",
        "flags",
        "synthetic",
    ]
    outbox = (
        DeliveryOutbox(destination / "delivery.sqlite", client, config.outbox_capacity) if client else None
    )
    failure = None
    status = "completed"
    try:
        with (
            (destination / "trajectories.csv").open("w", newline="") as tf,
            (destination / "events.jsonl").open("w") as ef,
            (destination / "features.jsonl").open("w") as ff,
            (destination / "event_geometry.jsonl").open("w") as gf,
            (destination / "frame_timestamps.csv").open("w", newline="") as ft,
        ):
            csv_writer = csv.DictWriter(tf, fieldnames=fields)
            csv_writer.writeheader()
            time_writer = csv.writer(ft)
            time_writer.writerow(
                ["frame", "timestamp_s", "timestamp_source", "discontinuity", "raw_timestamp_s"]
            )
            for frame in reader:
                tick = time.perf_counter()
                current_geometry = {
                    "width": frame.image.shape[1],
                    "height": frame.image.shape[0],
                    "live": reader.live,
                    "image_transform": "decoded_native",
                    "camera_pose_version": config.camera_pose_version,
                }
                if geometry is not None and current_geometry != geometry:
                    raise ValueError("Acquisition geometry changed; start a new calibrated run")
                if manifest is None:
                    geometry = current_geometry
                    if plane:
                        plane.validate_binding(
                            config.camera_id,
                            geometry["width"],
                            geometry["height"],
                            config.camera_pose_version,
                        )
                    manifest = build_manifest(
                        config,
                        resolved_source,
                        session_id,
                        "provided_capture_time" if video_start else "ingestion_start_not_capture_time",
                        geometry,
                        detector_override is not None,
                        wall_origin=origin,
                    )
                    temporary = destination / "run_manifest.json.tmp"
                    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False))
                    temporary.replace(destination / "run_manifest.json")
                decode_latencies.append(frame.decode_duration_s)
                if first_timestamp is None:
                    first_timestamp = frame.timestamp
                    first_source_frame = frame.index
                last_source_frame = frame.index
                last_timestamp = frame.timestamp
                coverage.observe(frame.timestamp, frame.discontinuity)
                time_writer.writerow(
                    [
                        frame.index,
                        frame.timestamp,
                        frame.timestamp_source,
                        frame.discontinuity,
                        frame.raw_timestamp_s,
                    ]
                )
                if frame.discontinuity:
                    motion.clear()
                    identity.clear()
                    detector.reset()
                    last_events.clear()
                    tracking_epoch += 1
                identity.begin_frame(frame.timestamp)
                inference_start = time.perf_counter()
                detections = detector.detect(frame.image, frame.timestamp)
                inference_latencies.append(time.perf_counter() - inference_start)
                detections_by_id = {d.track_id: d for d in detections if d.track_id is not None}
                current = []
                for detection in detections:
                    if detection.track_id is None:
                        continue
                    tid = detection.track_id
                    image_point = detection.position
                    ground = plane.project(image_point) if plane else None
                    if plane and ground is None:
                        outside_hull += 1
                        if config.calibration_fallback == "exclude":
                            continue
                    point, units = (ground, "m") if ground is not None else (image_point, "px")
                    previous = motion.previous.get(tid)
                    if (
                        previous is None
                        or frame.timestamp - previous.timestamp > motion.max_gap_s
                        or previous.units != units
                    ):
                        track_segments += 1
                    state = motion.update(tid, frame.timestamp, point, detection.confidence, units)
                    current.append((state, detection.class_name))
                    csv_writer.writerow(
                        dict(
                            zip(
                                fields,
                                [
                                    frame.index,
                                    frame.timestamp,
                                    frame.timestamp_source,
                                    tid,
                                    tracking_epoch,
                                    detection.class_name,
                                    detection.confidence,
                                    *detection.bbox,
                                    *image_point,
                                    *state.position,
                                    *state.velocity,
                                    *state.acceleration,
                                    units,
                                    state.quality,
                                    state.residual,
                                    "|".join(state.flags),
                                    config.synthetic,
                                ],
                            )
                        )
                    )
                    if config.save_video or preview is not None:
                        x1, y1, x2, y2 = map(int, detection.bbox)
                        cv2.rectangle(frame.image, (x1, y1), (x2, y2), (240, 180, 0), 2)
                        label = f"{detection.class_name} #{tid} {np.linalg.norm(state.velocity):.1f} {units}/s Q={state.quality:.2f}"
                        cv2.putText(frame.image, label, (x1, max(15, y1 - 8)), 0, 0.4, (255, 255, 255), 1)
                        # Ground-plane histories cannot be drawn as pixels; invert the homography explicitly.
                        path = np.array([p[1] for p in motion.history[tid]])
                        if units == "m" and plane:
                            path = cv2.perspectiveTransform(
                                path.reshape(-1, 1, 2), np.linalg.inv(plane.h)
                            ).reshape(-1, 2)
                        if len(path) > 1:
                            cv2.polylines(frame.image, [path.astype("int32")], False, (0, 255, 255), 2)
                scoring_started = time.perf_counter()
                highest = 0.0
                frame_scored = frame_abstained = 0
                for (a, class_a), (b, class_b) in combinations(current, 2):
                    if a.units != b.units:
                        mixed_coordinate_pairs += 1
                        continue
                    features = pair_features(
                        a,
                        b,
                        config.horizon_s,
                        calibration_uncertainty=plane.uncertainty_m if plane and a.units == "m" else 0,
                    )
                    # Unknown physical footprints: pipeline intentionally leaves disk TTC undefined.
                    proximity = config.proximity_m if a.units == "m" else config.proximity_px
                    assessment = identity.assess(
                        detections_by_id[a.track_id], detections_by_id[b.track_id], frame.timestamp
                    )
                    if assessment.ambiguous:
                        identity_abstentions += 1
                    risk = estimate(
                        features,
                        config.horizon_s,
                        proximity,
                        config.min_quality,
                        identity_ambiguous=assessment.ambiguous,
                    )
                    if config.pair_identity.enabled:
                        risk.model_version = "rules-0.1+identity-0.1"
                    projection = pair_projection(a, b, features.time_to_cpa_s, config.horizon_s)
                    projection["scoreable"] = not risk.abstained
                    projection["reliability_reasons"] = risk.reasons if risk.abstained else []
                    ff.write(
                        json.dumps(
                            {
                                "timestamp_s": frame.timestamp,
                                "tracking_epoch": tracking_epoch,
                                "track_ids": [a.track_id, b.track_id],
                                "features": features.dict(),
                                "risk_score": risk.score,
                                "abstained": risk.abstained,
                                "reasons": risk.reasons,
                                "synthetic": config.synthetic,
                                "pair_identity": assessment.measurements,
                                "projection": projection,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    if risk.abstained:
                        abstained += 1
                        frame_abstained += 1
                        continue
                    frame_scored += 1
                    highest = max(highest, risk.score)
                    pair = tuple(sorted([a.track_id, b.track_id]))
                    if (
                        risk.score < config.alert_threshold
                        or frame.timestamp - last_events.get(pair, -1e9) < config.cooldown_s
                    ):
                        continue
                    event = TrafficEvent(
                        session_id=session_id,
                        tracking_epoch=tracking_epoch,
                        camera_id=config.camera_id,
                        location_id=config.location_id,
                        timestamp=origin + timedelta(seconds=frame.timestamp),
                        timestamp_origin=manifest["content"]["timestamp_origin"],
                        timestamp_source=frame.timestamp_source,
                        raw_video_time_s=frame.raw_timestamp_s
                        if frame.raw_timestamp_s is not None and frame.raw_timestamp_s >= 0
                        else None,
                        run_manifest=manifest,
                        arrival_to_alert_ms=1000 * max(0, time.monotonic() - frame.acquired_monotonic_s)
                        if frame.acquired_monotonic_s is not None
                        else None,
                        video_time_s=frame.timestamp,
                        event_type="predicted_conflict",
                        risk_score=risk.score,
                        model_version=risk.model_version,
                        track_ids=[a.track_id, b.track_id],
                        classes=[class_a, class_b],
                        features=features.dict(),
                        explanations=risk.reasons,
                        synthetic=config.synthetic,
                    )
                    ef.write(event.model_dump_json() + "\n")
                    ef.flush()
                    gf.write(json.dumps({"event_id": str(event.event_id), "projection": projection}) + "\n")
                    gf.flush()
                    if outbox and not outbox.enqueue(event):
                        submission_failures += 1
                        log.error(
                            "delivery_queue_full; event retained for replay event_id=%s", event.event_id
                        )
                    last_events[pair] = frame.timestamp
                    emitted += 1
                scoring_latencies.append(time.perf_counter() - scoring_started)
                last_events = {
                    k: v for k, v in last_events.items() if frame.timestamp - v < 2 * config.cooldown_s
                }
                if config.save_video or preview is not None:
                    value = f"{highest:.2f}" if frame_scored else "unavailable"
                    title = f"{'SYNTHETIC | ' if config.synthetic else ''}Research heuristic max={value} | abstained={frame_abstained}"
                    cv2.rectangle(frame.image, (0, 0), (frame.image.shape[1], 46), (35, 25, 15), -1)
                    cv2.putText(frame.image, title, (10, 29), 0, 0.6, (255, 255, 255), 2)
                    if config.save_video:
                        if writer is None:
                            h, w = frame.image.shape[:2]
                            writer = cv2.VideoWriter(
                                str(destination / "annotated.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                reader.fps,
                                (w, h),
                            )
                            if not writer.isOpened():
                                raise RuntimeError("Output video encoder unavailable")
                        writer.write(frame.image)
                    if preview is not None:
                        preview.update(frame.image)
                if frame.acquired_monotonic_s is not None:
                    arrival_ages.append(max(0, time.monotonic() - frame.acquired_monotonic_s))
                frames += 1
                latencies.append(time.perf_counter() - tick)
                if realtime and not reader.live:
                    # Pace files for viewing; never invent/delay motion timestamps.
                    target = (frame.timestamp - first_timestamp) + 1 / reader.fps
                    time.sleep(max(0.0, target - (time.perf_counter() - started)))
                if config.max_frames and frames >= config.max_frames:
                    break
    except BaseException as exc:
        failure = exc
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    finally:
        reader.cap.release()
        if writer:
            writer.release()
    delivery = outbox.finish(config.delivery_shutdown_s) if outbox else None
    coverage_counts = {"complete": 0, "censored": 0}
    with (
        (destination / "events.jsonl").open() as events_file,
        (destination / "event_evaluation.jsonl.tmp").open("w") as evaluation_file,
    ):
        for line in events_file:
            event_data = json.loads(line)
            assessment = coverage.assess(event_data["video_time_s"], config.horizon_s)
            assessment["event_id"] = event_data["event_id"]
            coverage_counts["complete" if assessment["future_observation_complete"] else "censored"] += 1
            evaluation_file.write(json.dumps(assessment, allow_nan=False) + "\n")
    (destination / "event_evaluation.jsonl.tmp").replace(destination / "event_evaluation.jsonl")
    emitted = sum(coverage_counts.values())
    elapsed = time.perf_counter() - started
    report = {
        "status": status,
        "error_type": type(failure).__name__ if failure else None,
        "session_id": str(session_id),
        "camera_id": config.camera_id,
        "location_id": config.location_id,
        "synthetic": config.synthetic,
        "detector": config.detector,
        "detector_injected_for_test": detector_override is not None,
        "perception_diagnostics": getattr(detector, "diagnostics", {}),
        "config": config.model_dump(),
        "live_preview_enabled": preview is not None,
        "file_playback_paced": realtime and not reader.live,
        "frames": frames,
        "source_first_frame": first_source_frame,
        "source_last_frame": last_source_frame,
        "events": emitted,
        "abstained_pair_observations": abstained,
        "identity_ambiguity_abstentions": identity_abstentions,
        "observation_coverage": {
            "max_gap_s": config.observation_max_gap_s,
            "segments": len(coverage.segments),
            "event_horizons": coverage_counts,
        },
        "submission_failures": submission_failures + (delivery["failed_batches"] if delivery else 0),
        "delivery": delivery,
        "run_manifest_sha256": manifest["sha256"] if manifest else None,
        "calibration_outside_hull_observations": outside_hull,
        "mixed_coordinate_pair_exclusions": mixed_coordinate_pairs,
        "decode_latency_p95_ms": float(np.percentile(decode_latencies, 95) * 1000)
        if decode_latencies
        else None,
        "inference_latency_p95_ms": float(np.percentile(inference_latencies, 95) * 1000)
        if inference_latencies
        else None,
        "arrival_to_processed_p95_ms": float(np.percentile(arrival_ages, 95) * 1000)
        if arrival_ages
        else None,
        "scoring_and_event_metadata_p95_ms": float(np.percentile(scoring_latencies, 95) * 1000)
        if scoring_latencies
        else None,
        "capture_to_alert_ms": None,
        "live_timing_limit": "Capture timestamp/age unavailable; monotonic decode-arrival is not device capture time",
        "tracking_epochs": tracking_epoch + 1,
        "track_segments_not_validated_traffic_volume": track_segments,
        "latency_sample_count": len(latencies),
        "latency_sampling": "Most recent up to 10000 processed frames; excludes playback pacing",
        "video_duration_s": last_timestamp - (first_timestamp or 0),
        "elapsed_s": elapsed,
        "throughput_fps": frames / elapsed if elapsed else 0,
        "processing_latency_p50_ms": float(np.percentile(latencies, 50) * 1000) if latencies else None,
        "processing_latency_p95_ms": float(np.percentile(latencies, 95) * 1000) if latencies else None,
        "cpu_time_s": time.process_time() - cpu_started,
        "rss_end_mb": psutil.Process().memory_info().rss / 2**20,
        "network_body_bytes_attempted": client.bytes_attempted - initial_network_bytes if client else 0,
        "network_measurement": "JSON bodies including retries; excludes HTTP/TLS overhead; shutdown snapshot may precede in-flight completion",
        "timestamp_origin": "provided_capture_time" if video_start else "ingestion_start_not_capture_time",
        "calibration_validation_rmse_m": plane.validation_rmse_m if plane else None,
        "safety": "Research estimates; not a certified collision avoidance system",
    }
    temporary = destination / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(destination / "report.json")
    if failure is not None:
        raise failure
    return report
