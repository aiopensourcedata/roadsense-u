# SPDX-License-Identifier: AGPL-3.0-only
"""Scene-level published-collision pilot, separate from pair-level near-miss validation.

Dataset onset/impact annotations are targets only. Fixed scheduled snapshots use
past/current detections, never the ground-truth impact area or future tracks.
"""

import csv
import hashlib
import json
import math
from bisect import bisect_left
from pathlib import Path

from roadsense.camera.video_reader import VideoReader
from roadsense.evaluation.coverage import ObservationCoverage
from roadsense.risk.classical_model import FEATURES


def audit_collision_source(path: Path, annotation: dict, tolerance_s: float = 0.15) -> dict:
    """Bind a published annotation to exact decoded frames, dimensions and media PTS."""
    times, fallback, dimensions = [], 0, set()
    for frame in VideoReader(str(path)):
        times.append(frame.timestamp)
        fallback += frame.timestamp_source != "media_pts"
        dimensions.add((frame.image.shape[1], frame.image.shape[0]))
    reasons = []
    published_frames = int(annotation["no_frames"])
    if len(times) == published_frames:
        count_convention = "frame_count"
    elif len(times) == published_frames + 1:
        count_convention = "last_zero_based_frame_index_consistent"
    else:
        count_convention = "unresolved_mismatch"
        reasons.append("published_frame_count_mismatch")
    if dimensions != {(int(annotation["width"]), int(annotation["height"]))}:
        reasons.append("published_dimensions_mismatch")
    if fallback:
        reasons.append("fallback_timing_not_admissible_for_annotation_binding")
    index = int(annotation["accident_frame"])
    onset = float(annotation["accident_time"])
    error = abs(times[index] - onset) if 0 <= index < len(times) and math.isfinite(onset) else None
    if error is None or error > tolerance_s:
        reasons.append("published_collision_frame_time_mismatch")
    return {
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "frames": len(times),
        "timestamps_s": times,
        "timestamp_fallback_frames": fallback,
        "published_no_frames": published_frames,
        "frame_count_convention": count_convention,
        "dimensions": sorted(dimensions),
        "published_accident_time_s": onset,
        "published_accident_frame": index,
        "collision_frame_pts_s": times[index] if error is not None else None,
        "collision_time_binding_error_s": error,
        "binding_tolerance_s": tolerance_s,
        "valid_binding": not reasons,
        "exclusions": reasons,
    }


def collision_snapshots(
    run: Path,
    annotation: dict,
    source_timestamps: list[float],
    nominal_times: tuple[float, ...] = (1.0, 4.0, 7.0),
    horizon_s: float = 3.0,
    max_lateness_s: float = 0.25,
) -> list[dict]:
    """Extract fixed snapshots; preserve unscoreable positives as abstentions.

    Labels mean the published scene collision onset is inside the future horizon.
    They do not label every pair, absence of near misses, or actor correctness.
    Source continuity does not certify visibility or annotation correctness.
    """
    report = json.loads((run / "report.json").read_text())
    if report["synthetic"] or report.get("detector_injected_for_test"):
        raise ValueError("Published real-data pilot requires actual non-synthetic detector output")
    coverage = ObservationCoverage(report["config"]["observation_max_gap_s"])
    for t in source_timestamps:
        coverage.observe(t)
    with (run / "frame_timestamps.csv").open() as stream:
        timeline = list(csv.DictReader(stream))
    if len(timeline) != report["frames"]:
        raise ValueError("Run timeline/frame count mismatch")
    times = [float(row["timestamp_s"]) for row in timeline]
    for row in timeline:
        index = int(row["frame"])
        if (
            index >= len(source_timestamps)
            or abs(float(row["timestamp_s"]) - source_timestamps[index]) > 1e-6
        ):
            raise ValueError("Source timing does not match inference frames")
    selected, counts = {}, {}
    for nominal in nominal_times:
        k = bisect_left(times, nominal)
        if k < len(times):
            selected[times[k]] = None
            counts[times[k]] = [0, 0]
    with (run / "features.jsonl").open() as stream:
        for line in stream:
            pair = json.loads(line)
            t = pair["timestamp_s"]
            if t not in selected:
                continue
            counts[t][0] += 1
            if pair["abstained"]:
                continue
            counts[t][1] += 1
            if pair["synthetic"]:
                raise ValueError("Synthetic pair in real reference")
            if pair["features"].get("units") != "px":
                raise ValueError("This frozen scene baseline requires pixel-space features")
            # Deterministic choice uses measured features only, not annotation boxes/times.
            key = (-pair["risk_score"], tuple(sorted(pair["track_ids"])))
            previous = selected[t]
            if previous is None or key < (-previous["risk_score"], tuple(sorted(previous["track_ids"]))):
                selected[t] = pair
    result = []
    onset = float(annotation["accident_time"])
    for nominal in nominal_times:
        k = bisect_left(times, nominal)
        t = times[k] if k < len(times) else None
        reason = None
        observed = coverage.assess(t, horizon_s) if t is not None else None
        if t is None:
            reason = "scheduled_snapshot_unobserved"
        elif t - nominal > max_lateness_s:
            reason = "scheduled_snapshot_too_late"
        elif t >= onset:
            reason = "at_or_after_published_collision"
        elif not observed["future_observation_complete"]:
            reason = observed["reason"]
        pair = selected[t] if t is not None else None
        result.append(
            {
                "clip_id": annotation["filename"],
                "source_id": annotation["source_id"],
                "partition": annotation["partition"],
                "nominal_time_s": nominal,
                "prediction_time_s": t,
                "published_accident_time_s": onset,
                "eligible": reason is None,
                "exclusion_reason": reason,
                "label": int(onset <= t + horizon_s) if reason is None else None,
                "future_observation_complete": observed["future_observation_complete"] if observed else False,
                "scoreable": pair is not None,
                "units": pair["features"]["units"] if pair else None,
                "rule_score": pair["risk_score"] if pair else None,
                "selected_pair_ids": pair["track_ids"] if pair else None,
                "reported_pairs": counts[t][0] if t is not None else 0,
                "scoreable_pairs": counts[t][1] if t is not None else 0,
                **{feature: pair["features"][feature] if pair else None for feature in FEATURES},
            }
        )
    return result
