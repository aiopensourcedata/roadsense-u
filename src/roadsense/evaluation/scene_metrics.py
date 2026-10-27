# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit abstention and source-cluster metrics for labeled scene snapshots."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def scene_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    probability: bool = False,
    cohort_clip_ids: list[str] | None = None,
) -> dict:
    """Missing predictions remain abstentions, never confirmed safe negatives.

    AP/Brier are snapshot-level on accepted observations only. Operating recall
    includes unscoreable positives as no-warning misses. Event warning fractions
    deduplicate clip IDs and separately expose clips with no evaluable snapshot.
    """
    y, p = frame.label.to_numpy(), np.asarray(scores, dtype=float)
    if p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all() or np.isinf(p).any():
        raise ValueError("Nonempty binary labeled snapshots and matching finite/abstained scores required")
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    accepted = np.isfinite(p)
    if probability and ((p[accepted] < 0).any() or (p[accepted] > 1).any()):
        raise ValueError("Probabilities must lie in [0,1]")
    alert = accepted & (p >= threshold)
    tp, fp = int(sum(alert & (y == 1))), int(sum(alert & (y == 0)))
    positives, negatives = int(sum(y == 1)), int(sum(y == 0))
    accepted_positive = int(sum(accepted & (y == 1)))
    positive_clips = set(frame.loc[y == 1, "clip_id"])
    detected = set(frame.loc[alert & (y == 1), "clip_id"])
    cohort = set(cohort_clip_ids) if cohort_clip_ids is not None else set(frame.clip_id)
    if not set(frame.clip_id) <= cohort:
        raise ValueError("Snapshot clip is absent from declared evaluation cohort")
    warnings = []
    for _, group in frame.loc[alert & (y == 1)].groupby("clip_id"):
        onset = group.published_accident_time_s.unique()
        if len(onset) != 1:
            raise ValueError("One published collision onset per clip is required")
        warnings.append(float(onset[0] - group.prediction_time_s.min()))
    return {
        "n_snapshots": len(frame),
        "n_source_groups": int(frame.source_id.nunique()),
        "positive_snapshots": positives,
        "negative_snapshots": negatives,
        "accepted_snapshots": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "alert_true_positives": tp,
        "alert_false_positives": fp,
        "scored_true_negatives": int(sum(accepted & ~alert & (y == 0))),
        "scored_false_negatives": int(sum(accepted & ~alert & (y == 1))),
        "abstained_positive_snapshots": int(sum(~accepted & (y == 1))),
        "abstained_negative_snapshots": int(sum(~accepted & (y == 0))),
        "alert_precision": tp / (tp + fp) if tp + fp else None,
        "operating_recall_including_abstained_positives": tp / positives if positives else None,
        "accepted_only_recall": tp / accepted_positive if accepted_positive else None,
        "negative_snapshot_alert_fraction": fp / negatives if negatives else None,
        "accepted_only_pr_auc_average_precision": (
            float(average_precision_score(y[accepted], p[accepted]))
            if len(np.unique(y[accepted])) == 2
            else None
        ),
        "accepted_only_brier": float(brier_score_loss(y[accepted], p[accepted]))
        if probability and accepted.any()
        else None,
        "threshold": threshold,
        "event_warning": {
            "valid_labeled_cohort_clips": len(cohort),
            "warned_collision_clips": len(detected),
            "clips_with_eligible_positive_snapshot": len(positive_clips),
            "clips_without_eligible_positive_snapshot": len(cohort - positive_clips),
            "all_valid_clip_warning_fraction": len(detected) / len(cohort) if cohort else None,
            "no_warning_clips_including_unobservable": sorted(cohort - detected),
            "median_warning_s_detected_only": float(np.median(warnings)) if warnings else None,
        },
    }


def bootstrap_operating_recall(
    frame: pd.DataFrame,
    predictions: dict[str, tuple[np.ndarray, float]],
    repeats: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired resampling of whole source groups; small pilots give unstable intervals."""
    groups = frame.source_id.to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        return {"available": False, "reason": "fewer_than_two_source_groups"}
    labels = frame.label.to_numpy()
    values = {name: [] for name in predictions}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        idx = np.concatenate(
            [np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique), replace=True)]
        )
        positives = labels[idx] == 1
        if not positives.any():
            continue
        for name, (scores, threshold) in predictions.items():
            p = np.asarray(scores, dtype=float)[idx]
            values[name].append(float(np.mean(np.isfinite(p[positives]) & (p[positives] >= threshold))))
    intervals = {
        name: [float(x) for x in np.quantile(v, [0.025, 0.975])] if v else None for name, v in values.items()
    }
    differences = {}
    baseline = "rules_fixed_0.45"
    if baseline in values:
        for name, v in values.items():
            if name != baseline and v:
                differences[name + "_minus_" + baseline] = [
                    float(x) for x in np.quantile(np.asarray(v) - values[baseline], [0.025, 0.975])
                ]
    return {
        "available": True,
        "unit": "original_video_source_not_physical_location",
        "source_groups": len(unique),
        "repeats": repeats,
        "seed": seed,
        "operating_recall_95_percentile": intervals,
        "paired_recall_difference_95_percentile": differences,
        "limitations": "Few source groups; descriptive exploratory intervals. Camera independence/duplicate resolution unverified.",
    }
