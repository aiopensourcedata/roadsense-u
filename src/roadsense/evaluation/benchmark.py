# SPDX-License-Identifier: AGPL-3.0-only
"""Reproducible baseline comparison over independently labeled prediction observations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from roadsense.evaluation.metrics import cluster_bootstrap_ap, early_warning, metrics, select_threshold
from roadsense.evaluation.splits import split_data
from roadsense.evaluation.validation import load_observations
from roadsense.motion.features import PairFeatures
from roadsense.risk.classical_model import ClassicalModel
from roadsense.risk.rules import estimate


def rule_scores(frame):
    """Runtime eligibility and rule, for the pixel-space benchmark contract.

    Legacy tables without uncertainty/identity columns explicitly assume zero
    residual uncertainty and unambiguous identity; new studies should supply both.
    Missing measurements remain NaN, never an invented safe score.
    """
    scores = np.full(len(frame), np.nan)
    for i, row in enumerate(frame.to_dict("records")):
        if not row.get("scoreable", True):
            continue
        values = [row[k] for k in ("cpa_distance", "time_to_cpa_s", "closing_speed", "quality")]
        uncertainty = row.get("measurement_uncertainty", 0.0)
        if not np.isfinite([*values, uncertainty]).all():
            continue
        features = PairFeatures(
            distance=0,
            relative_speed=0,
            closing_speed=row["closing_speed"],
            cpa_distance=row["cpa_distance"],
            time_to_cpa_s=row["time_to_cpa_s"],
            ttc_s=None,
            shared_time_a_s=None,
            shared_time_b_s=None,
            arrival_difference_s=None,
            quality=row["quality"],
            measurement_uncertainty=uncertainty,
            braking_a=0,
            braking_b=0,
            direction_change_a_deg=0,
            direction_change_b_deg=0,
            units="px",
        )
        result = estimate(features, identity_ambiguous=row.get("identity_ambiguous", False))
        if not result.abstained:
            scores[i] = result.score
    return scores


def predict_available(model, frame):
    """Use the same measurement eligibility as the runtime rule for rescoring."""
    scores = np.full(len(frame), np.nan)
    mask = np.isfinite(rule_scores(frame))
    if mask.any():
        scores[mask] = model.predict(frame.loc[mask])
    return scores


def operating_metrics(labels, scores, threshold, probability=True):
    """Accepted-only AP/Brier, with all unavailable positives in recall denominator."""
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all() or np.isinf(p).any():
        raise ValueError("Matching nonempty binary labels and finite/abstained scores required")
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    mask = np.isfinite(p)
    result = (
        metrics(y[mask], p[mask], threshold, probability)
        if mask.any()
        else {
            "pr_auc_average_precision": None,
            "brier_score": None,
            "reliability_bins": [],
        }
    )
    alerts = mask & (p >= threshold)
    tp = int(sum(alerts & (y == 1)))
    result.update(
        {
            "precision": tp / int(alerts.sum()) if alerts.any() else 0.0,
            "recall": tp / int(sum(y == 1)) if (y == 1).any() else 0.0,
            "false_positive_event_fraction": float(alerts[y == 0].mean()) if (y == 0).any() else None,
            "n_events": len(y),
            "accepted_events": int(mask.sum()),
            "abstained_positive_events": int(sum(~mask & (y == 1))),
            "abstained_negative_events": int(sum(~mask & (y == 0))),
            "coverage": float(mask.mean()),
            "threshold": threshold,
            "ranking_and_calibration_scope": "accepted_only",
        }
    )
    return result


def event_table(frame, scores):
    """Aggregate each evaluation episode; negatives must be fixed-duration episodes."""
    data = frame.assign(score=scores)
    if (data.groupby("event_id").label.nunique() > 1).any():
        raise ValueError("Each evaluation episode must have a single binary horizon target")
    return data.groupby("event_id", as_index=False).agg(
        label=("label", "max"), score=("score", "max"), location_id=("location_id", "first")
    )


def benchmark(csv_path: str, output: str, split_mode: str = "location", seed: int = 42):
    frame = load_observations(csv_path)
    split = split_data(frame, split_mode, seed)
    parts = {k: frame.iloc[v].copy() for k, v in split.items()}
    model = ClassicalModel().fit(
        parts["train"].loc[np.isfinite(rule_scores(parts["train"]))],
        parts["calibration"].loc[np.isfinite(rule_scores(parts["calibration"]))],
    )
    seen = pd.concat([parts[k] for k in ("train", "calibration", "validation")])
    model.used_event_ids = set(seen.event_id.astype(str))
    model.used_clip_ids = set(seen.clip_id.astype(str))
    model.used_location_ids = set(seen.location_id.astype(str)) if split_mode == "location" else set()
    report = {
        "seed": seed,
        "observation_coverage": frame.attrs["observation_coverage"],
        "split_mode": split_mode,
        "synthetic": bool(frame.synthetic.astype(str).str.lower().eq("true").all()),
        "target": "labeled conflict in next 3 seconds, pre-event information only",
        "eligibility": "Shared runtime rules; absent uncertainty/identity columns assume zero/unambiguous (legacy contract)",
        "feature_units": "px and seconds; do not pool unnormalized metric/pixel locations",
        "partitions": {
            k: {
                "rows": len(v),
                "locations": sorted(v.location_id.unique().tolist()),
                "event_ids": sorted(v.event_id.astype(str).unique().tolist()),
            }
            for k, v in parts.items()
        },
        "baselines": {},
        "limitations": [
            "Synthetic fixtures establish no real-world performance.",
            "Range checking is a heuristic; uncertainty is not coverage guaranteed.",
            "False positive fraction is per negative episode, not false alerts per camera-hour.",
            "Location bootstrap intervals with few test locations are unstable.",
        ],
    }
    for name, predict, probability in [
        ("rules", rule_scores, False),
        ("centralized_logistic", lambda data: predict_available(model, data), True),
    ]:
        validation = event_table(parts["validation"], predict(parts["validation"]))
        available_validation = validation[validation.score.notna()]
        threshold = select_threshold(available_validation.label, available_validation.score, 0.05)
        p = predict(parts["test"])
        test = event_table(parts["test"], p)
        result = operating_metrics(test.label, test.score, threshold, probability)
        result["ap_95_cluster_bootstrap"] = cluster_bootstrap_ap(
            test.loc[test.score.notna(), "label"],
            test.loc[test.score.notna(), "score"],
            test.loc[test.score.notna(), "location_id"],
            seed,
        )
        result["early_warning"] = early_warning(parts["test"], p, threshold)
        result["recall_at_validation_selected_5pct_negative_episode_fpr"] = result["recall"]
        if probability:
            mask = ~np.isfinite(p) | model.out_of_range(parts["test"])
            result["range_abstention_fraction"] = float(mask.mean())
            if (~mask).any():
                accepted = event_table(parts["test"].iloc[np.flatnonzero(~mask)], p[~mask])
                result["accepted_only"] = metrics(accepted.label, accepted.score, threshold, True)
        report["baselines"][name] = result
    # Separate event splits within each camera/location: cannot imply unseen-location performance.
    local = {}
    for location, group in frame.groupby("location_id"):
        group = group.reset_index(drop=True)
        try:
            local_split = split_data(group, "event", seed)
            local_parts = {k: group.iloc[v] for k, v in local_split.items()}
            m = ClassicalModel().fit(
                local_parts["train"].loc[np.isfinite(rule_scores(local_parts["train"]))],
                local_parts["calibration"].loc[np.isfinite(rule_scores(local_parts["calibration"]))],
            )
            val = event_table(local_parts["validation"], predict_available(m, local_parts["validation"]))
            val = val[val.score.notna()]
            threshold = select_threshold(val.label, val.score)
            test = event_table(local_parts["test"], predict_available(m, local_parts["test"]))
            local[str(location)] = operating_metrics(test.label, test.score, threshold)
        except ValueError as exc:
            local[str(location)] = {"unavailable": str(exc)}
    report["independent_location_models_within_location_event_splits"] = local
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(str(directory / "baseline.joblib"))
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
