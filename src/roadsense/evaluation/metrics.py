# SPDX-License-Identifier: AGPL-3.0-only
"""Event metrics and cluster bootstrap; no frame-independence confidence intervals."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score


def _validated_arrays(labels, scores):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Scores and labels require nonempty equal-length 1D arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Labels must be binary and scores finite")
    return y, p


def select_threshold(labels, scores, max_false_positive_rate: float = 0.05) -> float:
    """Select on validation negatives only; applied unchanged to held-out test data."""
    y, p = _validated_arrays(labels, scores)
    if not 0 <= max_false_positive_rate <= 1:
        raise ValueError("False-positive rate must lie in [0,1]")
    if not (y == 0).any():
        raise ValueError("Negative validation events required")
    for threshold in np.unique(np.r_[np.nextafter(p.max(), np.inf), p]):
        if np.mean(p[y == 0] >= threshold) <= max_false_positive_rate:
            return float(threshold)
    return float(np.nextafter(p.max(), np.inf))


def metrics(labels, scores, threshold: float, probability: bool = True) -> dict:
    y, p = _validated_arrays(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    if probability and ((p < 0).any() or (p > 1).any()):
        raise ValueError("Probabilities must lie in [0,1]")
    alert = p >= threshold
    bins = []
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        chosen = (p >= left) & ((p < right) if right < 1 else (p <= right))
        if chosen.any():
            bins.append(
                {
                    "count": int(chosen.sum()),
                    "mean_score": float(p[chosen].mean()),
                    "observed_fraction": float(y[chosen].mean()),
                }
            )
    return {
        "precision": float(precision_score(y, alert, zero_division=0)),
        "recall": float(recall_score(y, alert, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y, p)) if (y == 1).any() else None,
        "brier_score": float(brier_score_loss(y, p)) if probability else None,
        "false_positive_event_fraction": float(alert[y == 0].mean()) if (y == 0).any() else None,
        "threshold": threshold,
        "reliability_bins": bins if probability else [],
        "n_events": len(y),
    }


def cluster_bootstrap_ap(labels, scores, groups, seed: int = 42, repeats: int = 200):
    y, p, g = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    unique = np.unique(g)
    if len(unique) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        idx = np.concatenate([np.flatnonzero(g == k) for k in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[idx])) == 2:
            estimates.append(average_precision_score(y[idx], p[idx]))
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])] if estimates else None


def early_warning(frame, scores, threshold: float):
    """First alert among evaluated pre-event observations; misses remain explicit."""
    values = []
    missed = 0
    temp = frame.assign(score=scores)
    positives = temp[temp.label == 1]
    for _, group in positives.groupby("event_id"):
        event_times = group.event_time_s.dropna().unique()
        if len(event_times) != 1:
            raise ValueError("One observed event time required for positive event")
        event_time = float(event_times[0])
        qualifying = group[(group.score >= threshold) & (group.prediction_time_s < event_time)]
        if qualifying.empty:
            missed += 1
        else:
            values.append(event_time - float(qualifying.prediction_time_s.min()))
    return {
        "detected_events": len(values),
        "missed_events": missed,
        "median_warning_s_detected_only": float(np.median(values)) if values else None,
    }


def approximate_cost(cpu_seconds: float, network_bytes: int, cpu_hour_price: float, gb_price: float) -> dict:
    """User supplied tariff scenario, not a cloud bill or a provider quote."""
    if (
        not np.isfinite([cpu_seconds, network_bytes, cpu_hour_price, gb_price]).all()
        or min(cpu_seconds, network_bytes, cpu_hour_price, gb_price) < 0
    ):
        raise ValueError("Cost measurements and tariffs must be nonnegative and finite")
    return {
        "scenario_cost": cpu_seconds / 3600 * cpu_hour_price + network_bytes / 1e9 * gb_price,
        "excludes": "idle capacity, storage, database, requests, TLS overhead, taxes",
    }
