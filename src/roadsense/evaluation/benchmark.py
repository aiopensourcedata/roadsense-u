# SPDX-License-Identifier: AGPL-3.0-only
"""Reproducible baseline comparison over independently labeled prediction observations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from roadsense.evaluation.metrics import cluster_bootstrap_ap, early_warning, metrics, select_threshold
from roadsense.evaluation.splits import split_data
from roadsense.evaluation.validation import load_observations
from roadsense.risk.classical_model import ClassicalModel


def rule_scores(frame):
    # Matches pipeline rule with a fixed 60px proximity scale for this pixel-space fixture.
    return (
        np.exp(-0.5 * (frame.cpa_distance.to_numpy() / 60) ** 2)
        * np.maximum(0, 1 - frame.time_to_cpa_s.to_numpy() / 3)
        * (frame.closing_speed.to_numpy() > 0)
    )


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
    model = ClassicalModel().fit(parts["train"], parts["calibration"])
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
        ("centralized_logistic", model.predict, True),
    ]:
        validation = event_table(parts["validation"], predict(parts["validation"]))
        threshold = select_threshold(validation.label, validation.score, 0.05)
        p = predict(parts["test"])
        test = event_table(parts["test"], p)
        result = metrics(test.label, test.score, threshold, probability)
        result["ap_95_cluster_bootstrap"] = cluster_bootstrap_ap(
            test.label, test.score, test.location_id, seed
        )
        result["early_warning"] = early_warning(parts["test"], p, threshold)
        result["recall_at_validation_selected_5pct_negative_episode_fpr"] = result["recall"]
        if probability:
            mask = model.out_of_range(parts["test"])
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
            m = ClassicalModel().fit(local_parts["train"], local_parts["calibration"])
            val = event_table(local_parts["validation"], m.predict(local_parts["validation"]))
            threshold = select_threshold(val.label, val.score)
            test = event_table(local_parts["test"], m.predict(local_parts["test"]))
            local[str(location)] = metrics(test.label, test.score, threshold)
        except ValueError as exc:
            local[str(location)] = {"unavailable": str(exc)}
    report["independent_location_models_within_location_event_splits"] = local
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(str(directory / "baseline.joblib"))
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
