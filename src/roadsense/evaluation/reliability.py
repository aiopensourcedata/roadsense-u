# SPDX-License-Identifier: AGPL-3.0-only
"""Paired causal reliability ablations on identical cached detections/features.

These policies assess abstention and alert stability, not conflict accuracy. New
policies remain offline experiments and do not alter the production pipeline.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations, groupby
from pathlib import Path

from roadsense.cloud.schemas import FeatureSummary, TrafficEvent
from roadsense.config import PairIdentityConfig, PipelineConfig
from roadsense.detection.pair_identity import PairIdentityGuard
from roadsense.detection.tracker import Detection
from roadsense.evaluation.coverage import ObservationCoverage
from roadsense.motion.features import PairFeatures
from roadsense.motion.trajectories import Trajectories
from roadsense.risk.rules import estimate

POLICIES = (
    "geometry",
    "identity_current",
    "identity_pending",
    "identity_history_0_5",
    "identity_pending_history_0_5",
)


def policy_acceptance(
    scoreable: bool,
    ambiguous: bool,
    measurements: dict,
    minimum_track_age_s: float,
    config: PairIdentityConfig,
) -> dict[str, bool]:
    """Fixed experimental policies; use only currently available evidence.

    Pending overlap is an abstention, never a duplicate classification. Distinct
    occluding objects may lose coverage. Ages refer to contiguous track segments.
    """
    if not math.isfinite(minimum_track_age_s) or minimum_track_age_s < 0:
        raise ValueError("Track age must be nonnegative and finite")
    pending = (
        measurements.get("iou", 0) >= config.min_iou
        and measurements.get("center_separation_box_diagonals", math.inf) <= config.max_center_separation
        and (
            measurements.get("samples", 0) < config.min_samples
            or measurements.get("span_s", 0) < config.min_span_s
        )
    )
    current = scoreable and not ambiguous
    history = minimum_track_age_s >= 0.5 - 1e-9
    return {
        "geometry": scoreable,
        "identity_current": current,
        "identity_pending": current and not pending,
        "identity_history_0_5": current and history,
        "identity_pending_history_0_5": current and not pending and history,
    }


def _signature(events: list[dict]) -> list[tuple]:
    return sorted(
        (
            round(e["video_time_s"], 8),
            e["tracking_epoch"],
            tuple(sorted(e["track_ids"])),
            round(e["risk_score"], 10),
        )
        for e in events
    )


def compare_reliability(reference: Path, output: Path) -> dict:
    """Recompute rules and reproduce reference decisions before reporting ablations.

    Complete frame timelines are required, including empty frames. Both guarded
    and unguarded reference runs are supported. Identity/track ages are rebuilt
    from past/current boxes; observation coverage is retrospective and never used
    for inference. Replay timing is not video-processing throughput.
    """
    if output.exists():
        raise ValueError("Output already exists; use a fresh directory")
    names = ["report.json", "trajectories.csv", "features.jsonl", "frame_timestamps.csv", "events.jsonl"]
    hashes = {name: hashlib.sha256((reference / name).read_bytes()).hexdigest() for name in names}
    report = json.loads((reference / "report.json").read_text())
    cfg = PipelineConfig(**report["config"])
    identity_cfg = cfg.pair_identity.model_copy(update={"enabled": True})
    guard, coverage = PairIdentityGuard(identity_cfg), ObservationCoverage(cfg.observation_max_gap_s)
    max_gap = Trajectories().max_gap_s
    counts = {name: Counter(accepted=0, high_score_observations=0) for name in POLICIES}
    alerts: dict[str, list] = {name: [] for name in POLICIES}
    last = {name: {} for name in POLICIES}
    track_history = {}
    pairs = defaultdict(lambda: {"observations": 0, "accepted": Counter()})
    changes = []
    total = frames = epoch = 0
    reference_policy = "identity_current" if cfg.pair_identity.enabled else "geometry"
    with (
        (reference / "trajectories.csv").open() as trajectories,
        (reference / "features.jsonl").open() as features,
        (reference / "frame_timestamps.csv").open() as timestamps,
    ):
        groups = iter(groupby(csv.DictReader(trajectories), key=lambda row: int(row["frame"])))
        group = next(groups, None)
        cached = (json.loads(line) for line in features)
        for frame in csv.DictReader(timestamps):
            index, timestamp = int(frame["frame"]), float(frame["timestamp_s"])
            if frame["discontinuity"] not in {"True", "False"}:
                raise ValueError("Frame discontinuity must be explicit")
            discontinuity = frame["discontinuity"] == "True"
            coverage.observe(timestamp, discontinuity)
            frames += 1
            if discontinuity:
                epoch += 1
                guard.clear()
                track_history.clear()
                last = {name: {} for name in POLICIES}
            guard.begin_frame(timestamp)
            rows = []
            if group is not None:
                if group[0] < index:
                    raise ValueError("Trajectory frames do not align with the complete timeline")
                if group[0] == index:
                    rows = list(group[1])
                    group = next(groups, None)
            detections, ages = {}, {}
            track_history = {k: v for k, v in track_history.items() if timestamp - v[1] <= max_gap}
            for row in rows:
                if int(row["tracking_epoch"]) != epoch or abs(float(row["timestamp_s"]) - timestamp) > 1e-8:
                    raise ValueError("Trajectory timestamp/epoch mismatch")
                tid = int(row["track_id"])
                if tid in detections:
                    raise ValueError("Repeated track ID within a frame")
                detections[tid] = Detection(
                    tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2")),
                    row["class"],
                    float(row["confidence"]),
                    tid,
                )
                previous = track_history.get(tid)
                start = previous[0] if previous and previous[2] == row["units"] else timestamp
                ages[tid] = timestamp - start
                track_history[tid] = (start, timestamp, row["units"])
            for a, b in combinations(rows, 2):
                if a["units"] != b["units"]:
                    continue
                original = next(cached, None)
                ids = [int(a["track_id"]), int(b["track_id"])]
                if (
                    original is None
                    or original["track_ids"] != ids
                    or abs(original["timestamp_s"] - timestamp) > 1e-8
                    or original["tracking_epoch"] != epoch
                ):
                    raise ValueError("Cached pair features do not match trajectories/timeline")
                if original.get("synthetic") != report["synthetic"]:
                    raise ValueError("Cached feature provenance disagrees with the run")
                measured = PairFeatures(**FeatureSummary.model_validate(original["features"]).model_dump())
                if measured.units != a["units"]:
                    raise ValueError("Cached measurement units disagree")
                risk = estimate(
                    measured,
                    cfg.horizon_s,
                    cfg.proximity_m if measured.units == "m" else cfg.proximity_px,
                    cfg.min_quality,
                )
                assessment = guard.assess(detections[ids[0]], detections[ids[1]], timestamp)
                decisions = policy_acceptance(
                    not risk.abstained,
                    assessment.ambiguous,
                    assessment.measurements,
                    min(ages[ids[0]], ages[ids[1]]),
                    identity_cfg,
                )
                accepted = decisions[reference_policy]
                if original["abstained"] != (not accepted):
                    raise ValueError("Reference abstention decisions could not be reproduced")
                if accepted:
                    if original["risk_score"] is None or not math.isclose(
                        risk.score, original["risk_score"], abs_tol=1e-10, rel_tol=1e-10
                    ):
                        raise ValueError("Reference risk scores could not be reproduced")
                elif original["risk_score"] is not None:
                    raise ValueError("Abstained reference contains a numeric risk score")
                if cfg.pair_identity.enabled and original.get("pair_identity") != assessment.measurements:
                    raise ValueError("Reference identity measurements could not be reproduced")
                key = (epoch, *sorted(ids))
                pair = pairs[key]
                pair["observations"] += 1
                total += 1
                if len(set(decisions.values())) > 1:
                    changes.append(
                        {
                            "frame": index,
                            "timestamp_s": timestamp,
                            "tracking_epoch": epoch,
                            "track_ids": ids,
                            "risk_score_before_identity": risk.score,
                            "minimum_track_age_s": min(ages[k] for k in ids),
                            "identity_measurements": assessment.measurements,
                            "accepted": decisions,
                        }
                    )
                for name, chosen in decisions.items():
                    counts[name]["accepted"] += chosen
                    pair["accepted"][name] += chosen
                    if not chosen or risk.score < cfg.alert_threshold:
                        continue
                    counts[name]["high_score_observations"] += 1
                    if timestamp - last[name].get(key, -1e9) < cfg.cooldown_s:
                        continue
                    alerts[name].append(
                        {
                            "video_time_s": timestamp,
                            "tracking_epoch": epoch,
                            "track_ids": ids,
                            "risk_score": risk.score,
                        }
                    )
                    last[name][key] = timestamp
            for name in POLICIES:
                last[name] = {k: t for k, t in last[name].items() if timestamp - t < 2 * cfg.cooldown_s}
        if group is not None or next(cached, None) is not None:
            raise ValueError("Unmatched cached trajectory/feature rows")
    reference_events = [
        TrafficEvent.model_validate_json(line).model_dump(mode="json")
        for line in (reference / "events.jsonl").read_text().splitlines()
    ]
    if frames != report["frames"] or _signature(reference_events) != _signature(alerts[reference_policy]):
        raise ValueError("Reference frame count or alert emissions could not be reproduced")
    if counts[reference_policy]["accepted"] != total - report["abstained_pair_observations"]:
        raise ValueError("Reference accepted-pair count could not be reproduced")
    if any(hashlib.sha256((reference / name).read_bytes()).hexdigest() != hashes[name] for name in names):
        raise ValueError("Reference artifacts changed during replay")
    for name in POLICIES:
        for event in alerts[name]:
            event["future_observation"] = coverage.assess(event["video_time_s"], cfg.horizon_s)
    result = {
        "mode": "paired_cached_reliability_ablation",
        "source_run": str(reference),
        "source_session_id": report["session_id"],
        "synthetic": report["synthetic"],
        "detector_injected_for_test": report.get("detector_injected_for_test", False),
        "source_sha256": hashes,
        "config": cfg.model_dump(),
        "reference_decisions_scores_alerts_reproduced": True,
        "frames": frames,
        "reported_pair_observations": total,
        "policies": {
            name: {
                **dict(counts[name]),
                "accepted_fraction": counts[name]["accepted"] / total if total else None,
                "emissions": len(alerts[name]),
                "alerts": alerts[name],
            }
            for name in POLICIES
        },
        "limitations": [
            "No independent conflict labels: no precision, recall or false-alarm improvement is inferred.",
            "Same-camera frame pairs are correlated; no frame-level confidence interval is reported.",
            "Offline abstention policies do not alter detector outputs, geometry or production policy.",
            "Minimum history/pending overlap can suppress genuine distinct-object interactions.",
            "Cached replay excludes decoding/inference and is not an end-to-end FPS benchmark.",
        ],
    }
    # Publish no successful result until reference integrity checks pass.
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "changes.jsonl").open("w") as stream:
        for row in changes:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (output / "pair_summary.jsonl").open("w") as stream:
        for (ep, a, b), row in sorted(pairs.items()):
            stream.write(json.dumps({"tracking_epoch": ep, "track_ids": [a, b], **row}) + "\n")
    return result
