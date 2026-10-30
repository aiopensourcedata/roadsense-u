# SPDX-License-Identifier: AGPL-3.0-only
"""Frozen source-disjoint collision-onset pilot; no physical-location claims."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from roadsense.evaluation.metrics import select_threshold
from roadsense.evaluation.scene_metrics import bootstrap_operating_recall, scene_metrics
from roadsense.risk.classical_model import FEATURES, ClassicalModel


def validate_scene_observations(frame: pd.DataFrame) -> None:
    """Reject cross-partition source reuse, missing outcomes and post-event leakage."""
    required = {
        "partition",
        "source_id",
        "clip_id",
        "nominal_time_s",
        "prediction_time_s",
        "published_accident_time_s",
        "label",
        "future_observation_complete",
        "scoreable",
        "units",
        "rule_score",
        *FEATURES,
    }
    if not required <= set(frame.columns):
        raise ValueError("Missing required scene observation columns")
    for key in ("source_id", "clip_id", "partition"):
        if (
            not frame[key]
            .map(lambda value: isinstance(value, str) and bool(value) and value == value.strip())
            .all()
        ):
            raise ValueError(f"Nonempty canonical {key} required")
    for key in ("scoreable", "future_observation_complete"):
        if not frame[key].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"{key} must contain actual boolean values")
    if (
        frame.groupby("clip_id").source_id.nunique().max() != 1
        or frame.groupby("clip_id").partition.nunique().max() != 1
    ):
        raise ValueError("Each clip must map to one source and partition")
    timing = frame[["nominal_time_s", "prediction_time_s", "published_accident_time_s"]].to_numpy(dtype=float)
    if not np.isfinite(timing).all() or (timing < 0).any():
        raise ValueError("Scene timestamps must be finite and nonnegative")
    if frame.groupby("clip_id").published_accident_time_s.nunique().max() != 1:
        raise ValueError("One published onset per clip is required")
    if frame.empty or set(frame.partition) != {"train", "calibration", "validation", "test"}:
        raise ValueError("Four nonempty frozen partitions are required")
    if frame.groupby("source_id").partition.nunique().max() != 1:
        raise ValueError("An original video source crosses partitions")
    if frame.duplicated(["clip_id", "nominal_time_s"]).any():
        raise ValueError("Duplicate scheduled snapshot")
    if not frame.label.isin([0, 1]).all() or not frame.future_observation_complete.eq(True).all():
        raise ValueError("Binary targets and complete future observation are required")
    if not (frame.prediction_time_s < frame.published_accident_time_s).all():
        raise ValueError("Post-event snapshots cannot enter training/evaluation")
    expected = (frame.published_accident_time_s <= frame.prediction_time_s + 3).astype(int)
    if not np.array_equal(expected, frame.label):
        raise ValueError("Target disagrees with published onset and three-second horizon")
    if frame.loc[~frame.scoreable, "rule_score"].notna().any():
        raise ValueError("Unavailable observations must not contain a rule score")
    accepted = frame[frame.scoreable]
    if not accepted.units.eq("px").all():
        raise ValueError("Scene baseline requires pixel units")
    if accepted[FEATURES + ["rule_score"]].isna().any().any():
        raise ValueError("Scoreable observations require actual complete feature measurements")
    if not np.isfinite(accepted[FEATURES + ["rule_score"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite scene measurements")


def run_scene_benchmark(
    frame: pd.DataFrame,
    cohort: list[dict],
    output: Path,
    seed: int = 42,
) -> dict:
    """Fit on train, calibrate separately, select thresholds on validation, then test.

    Candidate selection and absence of valid features are supplied by the frozen
    video extraction protocol. Missing scores remain abstentions. No model is
    invented when accepted training/calibration observations lack both classes.
    """
    if output.exists():
        raise ValueError("Output exists; choose a fresh experiment directory")
    validate_scene_observations(frame)
    mapping = {}
    source_partitions = {}
    for row in cohort:
        clip, partition = row.get("filename"), row.get("partition")
        if not isinstance(clip, str) or not clip or clip != clip.strip() or clip in mapping:
            raise ValueError("Cohort requires unique canonical clip IDs")
        if partition not in {"train", "calibration", "validation", "test"}:
            raise ValueError("Invalid cohort partition")
        source = row.get("source_id")
        if source is not None:
            if not isinstance(source, str) or not source or source != source.strip():
                raise ValueError("Invalid cohort source ID")
            if source in source_partitions and source_partitions[source] != partition:
                raise ValueError("Cohort source crosses partitions")
            source_partitions[source] = partition
        mapping[clip] = row
    for row in frame[["clip_id", "source_id", "partition"]].drop_duplicates().to_dict("records"):
        declared = mapping.get(row["clip_id"])
        if (
            declared is None
            or declared["partition"] != row["partition"]
            or declared.get("source_id", row["source_id"]) != row["source_id"]
        ):
            raise ValueError("Snapshot source/partition disagrees with cohort")
    parts = {name: frame[frame.partition == name].reset_index(drop=True) for name in frame.partition.unique()}
    test, validation = parts["test"], parts["validation"]
    cohort_ids = [row["filename"] for row in cohort if row["partition"] == "test"]
    rules = np.where(test.scoreable.to_numpy(dtype=bool), test.rule_score.to_numpy(dtype=float), np.nan)
    predictions = {"rules_fixed_0.45": (rules, 0.45)}
    probabilities = {"rules_fixed_0.45": False}
    # Explicit no-feature reference, not a replacement for missing detector output.
    prior = float(parts["train"].label.mean())
    prior_threshold = select_threshold(validation.label, np.full(len(validation), prior), 0.05)
    predictions["training_prior"] = (np.full(len(test), prior), prior_threshold)
    probabilities["training_prior"] = True
    matched_prior = np.full(len(test), np.nan)
    matched_prior[test.scoreable.to_numpy(dtype=bool)] = prior
    predictions["training_prior_matched_perception"] = (matched_prior, prior_threshold)
    probabilities["training_prior_matched_perception"] = True
    unavailable = {}
    val_scored = validation[validation.scoreable]
    try:
        threshold = select_threshold(val_scored.label, val_scored.rule_score, 0.05)
        predictions["rules_validation_threshold"] = (rules, threshold)
        probabilities["rules_validation_threshold"] = False
    except ValueError as exc:
        unavailable["rules_validation_threshold"] = str(exc)
    trained = None
    try:
        train = parts["train"][parts["train"].scoreable]
        calibration = parts["calibration"][parts["calibration"].scoreable]
        if train.label.nunique() != 2 or calibration.label.nunique() != 2:
            raise ValueError("Accepted training and calibration observations must each contain both classes")
        model = ClassicalModel().fit(train, calibration)
        values = np.full(len(test), np.nan)
        mask = test.scoreable.to_numpy(dtype=bool)
        range_values = values.copy()
        if mask.any():
            values[mask] = model.predict(test.loc[mask])
            range_values[mask] = values[mask]
            outside = model.out_of_range(test.loc[mask])
            range_values[np.flatnonzero(mask)[outside]] = np.nan
        trained = model
        predictions["calibrated_logistic_fixed_0.5"] = (values, 0.5)
        predictions["calibrated_logistic_range_fixed_0.5"] = (range_values, 0.5)
        probabilities.update(
            {
                "calibrated_logistic_fixed_0.5": True,
                "calibrated_logistic_range_fixed_0.5": True,
            }
        )
        try:
            if val_scored.empty:
                raise ValueError("No scoreable validation snapshots")
            threshold = select_threshold(val_scored.label, model.predict(val_scored), 0.05)
            predictions["calibrated_logistic_validation_threshold"] = (values, threshold)
            predictions["calibrated_logistic_range_validation_threshold"] = (range_values, threshold)
            probabilities.update(
                {
                    "calibrated_logistic_validation_threshold": True,
                    "calibrated_logistic_range_validation_threshold": True,
                }
            )
        except ValueError as exc:
            unavailable["calibrated_logistic_validation_threshold"] = str(exc)
    except ValueError as exc:
        unavailable["calibrated_logistic"] = str(exc)
    metrics = {
        name: scene_metrics(test, scores, threshold, probabilities[name], cohort_ids)
        for name, (scores, threshold) in predictions.items()
    }
    report = {
        "evaluation_kind": "published_scene_collision_pilot",
        "target": "Published scene collision onset in (prediction_timestamp, prediction_timestamp+3s]",
        "label_source": "Published ACCIDENT annotations, separate from model outputs; not independently re-adjudicated here",
        "synthetic": False,
        "split": "frozen_original_video_source_disjoint_not_location_disjoint",
        "seed": seed,
        "partitions": {
            name: {
                "snapshots": len(p),
                "source_groups": int(p.source_id.nunique()),
                "positive_snapshots": int(sum(p.label == 1)),
                "negative_snapshots": int(sum(p.label == 0)),
                "scoreable_snapshots": int(p.scoreable.sum()),
                "scoreable_positive_snapshots": int(sum(p.scoreable & (p.label == 1))),
                "scoreable_negative_snapshots": int(sum(p.scoreable & (p.label == 0))),
            }
            for name, p in parts.items()
        },
        "baselines": metrics,
        "unavailable_baselines": unavailable,
        "source_cluster_bootstrap": bootstrap_operating_recall(test, predictions, seed=seed),
        "limitations": [
            "Small collision-enriched convenience sample, not the official ACCIDENT benchmark split.",
            "Pre-event negative horizons are negative for the published onset target, not validated absence of near misses.",
            "No physical camera mapping or full cross-source near-duplicate audit: no held-out-camera/generalization claim.",
            "Scene maximum over scoreable pairs does not identify the true colliding actors; scoreable coverage can be partial.",
            "Accepted-only AP/Brier do not measure the full unscoreable cohort. Abstentions and event misses are explicit.",
            "Snapshots can be correlated/overlap slightly under variable PTS; whole original sources form bootstrap units.",
            "Few source clusters make intervals unstable; no Q1-readiness or novelty/improvement conclusion.",
        ],
    }
    output.mkdir(parents=True)
    if trained is not None:
        trained.prediction_target = "published_scene_collision_onset_next_3_media_seconds"
        trained.training_source_ids = sorted(parts["train"].source_id.unique().tolist())
        trained.calibration_source_ids = sorted(parts["calibration"].source_id.unique().tolist())
        trained.save(str(output / "scene-baseline.joblib"))
        (output / "model-card.json").write_text(
            json.dumps(
                {
                    "target": trained.prediction_target,
                    "features": FEATURES,
                    "units": "pixels and media seconds",
                    "selection": "Maximum current-policy scoreable pair; physical colliding actors are not verified",
                    "use": "Offline research scene pilot only; not integrated as a pair-level near-miss or safety model",
                    "training_sources": trained.training_source_ids,
                    "calibration_sources": trained.calibration_source_ids,
                    "calibration_scope": "Separate, small source cohort; no guarantee of calibrated deployment probabilities",
                    "artifact_loading": "Trusted local joblib only; never load untrusted model files",
                },
                indent=2,
            )
            + "\n"
        )
    frame.to_csv(output / "eligible-snapshots.csv", index=False)
    records = []
    for i, row in test.iterrows():
        records.append(
            {
                "clip_id": row.clip_id,
                "source_id": row.source_id,
                "prediction_time_s": row.prediction_time_s,
                "label": int(row.label),
                "predictions": {
                    name: {
                        "score": float(scores[i]) if np.isfinite(scores[i]) else None,
                        "threshold": threshold,
                    }
                    for name, (scores, threshold) in predictions.items()
                },
            }
        )
    (output / "test-predictions.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
