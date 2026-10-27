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
from roadsense.cloud.schemas import TrafficEvent
from roadsense.config import PipelineConfig
from roadsense.detection.detector import SyntheticDetector, YoloByteTrack
from roadsense.detection.pair_identity import PairIdentityGuard
from roadsense.evaluation.coverage import ObservationCoverage
from roadsense.motion.features import pair_features, pair_projection
from roadsense.motion.trajectories import Trajectories
from roadsense.risk.rules import RiskEstimate, estimate

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
    reader = VideoReader(parse_source(source), start_frame=config.start_frame)
    try:
        detector = detector_override or (
            SyntheticDetector()
            if config.detector == "synthetic"
            else YoloByteTrack(config.weights, config.confidence, config.device, config.inference_size)
        )
        plane = GroundPlane.load(config.calibration) if config.calibration else None
    except Exception:
        reader.cap.release()
        raise
    motion, session_id = Trajectories(), uuid4()
    identity = PairIdentityGuard(config.pair_identity)
    coverage = ObservationCoverage(config.observation_max_gap_s)
    identity_abstentions = 0
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
            time_writer.writerow(["frame", "timestamp_s", "timestamp_source", "discontinuity"])
            for frame in reader:
                tick = time.perf_counter()
                if first_timestamp is None:
                    first_timestamp = frame.timestamp
                    first_source_frame = frame.index
                last_source_frame = frame.index
                last_timestamp = frame.timestamp
                coverage.observe(frame.timestamp, frame.discontinuity)
                time_writer.writerow(
                    [frame.index, frame.timestamp, frame.timestamp_source, frame.discontinuity]
                )
                if frame.discontinuity:
                    motion.clear()
                    identity.clear()
                    detector.reset()
                    last_events.clear()
                    tracking_epoch += 1
                identity.begin_frame(frame.timestamp)
                detections = detector.detect(frame.image, frame.timestamp)
                detections_by_id = {d.track_id: d for d in detections if d.track_id is not None}
                current = []
                for detection in detections:
                    if detection.track_id is None:
                        continue
                    tid = detection.track_id
                    image_point = detection.position
                    ground = plane.project(image_point) if plane else None
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
                highest = 0.0
                frame_scored = frame_abstained = 0
                for (a, class_a), (b, class_b) in combinations(current, 2):
                    if a.units != b.units:
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
                        risk = RiskEstimate(None, True, ["ambiguous_pair_identity"])
                        identity_abstentions += 1
                    else:
                        risk = estimate(features, config.horizon_s, proximity, config.min_quality)
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
                    if client:
                        try:
                            client.submit(event)
                        except Exception as exc:
                            submission_failures += 1
                            log.error(
                                "event_submission_failed type=%s event_id=%s; metadata retained for replay",
                                type(exc).__name__,
                                event.event_id,
                            )
                    last_events[pair] = frame.timestamp
                    emitted += 1
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
                frames += 1
                latencies.append(time.perf_counter() - tick)
                if realtime and not reader.live:
                    # Pace files for viewing; never invent/delay motion timestamps.
                    target = (frame.timestamp - first_timestamp) + 1 / reader.fps
                    time.sleep(max(0.0, target - (time.perf_counter() - started)))
                if config.max_frames and frames >= config.max_frames:
                    break
    finally:
        reader.cap.release()
        if writer:
            writer.release()
    coverage_counts = {"complete": 0, "censored": 0}
    with (
        (destination / "events.jsonl").open() as events_file,
        (destination / "event_evaluation.jsonl").open("w") as evaluation_file,
    ):
        for line in events_file:
            event_data = json.loads(line)
            assessment = coverage.assess(event_data["video_time_s"], config.horizon_s)
            assessment["event_id"] = event_data["event_id"]
            coverage_counts["complete" if assessment["future_observation_complete"] else "censored"] += 1
            evaluation_file.write(json.dumps(assessment, allow_nan=False) + "\n")
    elapsed = time.perf_counter() - started
    report = {
        "session_id": str(session_id),
        "camera_id": config.camera_id,
        "location_id": config.location_id,
        "synthetic": config.synthetic,
        "detector": config.detector,
        "detector_injected_for_test": detector_override is not None,
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
        "submission_failures": submission_failures,
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
        "network_measurement": "JSON bodies including retries; excludes HTTP/TLS overhead",
        "timestamp_origin": "provided_capture_time" if video_start else "ingestion_start_not_capture_time",
        "calibration_validation_rmse_m": plane.validation_rmse_m if plane else None,
        "safety": "Research estimates; not a certified collision avoidance system",
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    return report
