# SPDX-License-Identifier: AGPL-3.0-only
"""Published-label binding, target timing, explicit abstention and leakage controls."""

import csv
import json

import cv2
import joblib
import numpy as np
import pandas as pd
import pytest

from roadsense.evaluation.collision_pilot import audit_collision_source, collision_snapshots
from roadsense.evaluation.scene_benchmark import run_scene_benchmark, validate_scene_observations
from roadsense.evaluation.scene_metrics import bootstrap_operating_recall, scene_metrics
from roadsense.risk.classical_model import FEATURES


def test_binding_records_count_conventions_and_rejects_bad_timing(tmp_path):
    source = tmp_path / "synthetic-audit.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 48))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((48, 80, 3), np.uint8))
    writer.release()
    metadata = {"no_frames": 19, "width": 80, "height": 48, "accident_frame": 8, "accident_time": 0.8}
    audit = audit_collision_source(source, metadata)
    assert (
        audit["valid_binding"] and audit["frame_count_convention"] == "last_zero_based_frame_index_consistent"
    )
    assert audit_collision_source(source, {**metadata, "no_frames": 20})["valid_binding"]
    assert not audit_collision_source(source, {**metadata, "no_frames": 18})["valid_binding"]
    assert not audit_collision_source(source, {**metadata, "accident_time": 1.3})["valid_binding"]


def cached_scene(tmp_path, timestamps=None, stop=11, empty=False):
    """Handcrafted unit fixture only; not an actual detector/performance result."""
    times = timestamps if timestamps is not None else [i / 10 for i in range(111)]
    run = tmp_path
    run.mkdir(exist_ok=True)
    observed = [(i, t) for i, t in enumerate(times) if t <= stop]
    (run / "report.json").write_text(
        json.dumps(
            {
                "synthetic": False,
                "detector_injected_for_test": False,
                "frames": len(observed),
                "config": {"observation_max_gap_s": 0.5},
            }
        )
    )
    with (run / "frame_timestamps.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["frame", "timestamp_s"])
        w.writerows(observed)
    features = []
    for t, score, ids in [(1, 0.1, [9, 10]), (4, 0.7, [4, 2]), (4, 0.7, [1, 3]), (7, 0.99, [5, 6])]:
        features.append(
            {
                "timestamp_s": t,
                "synthetic": False,
                "abstained": False,
                "risk_score": score,
                "track_ids": ids,
                "features": {
                    "cpa_distance": 50,
                    "time_to_cpa_s": 1,
                    "closing_speed": 2,
                    "quality": 0.9,
                    "units": "px",
                },
            }
        )
    (run / "features.jsonl").write_text("" if empty else "\n".join(json.dumps(r) for r in features) + "\n")
    annotation = {
        "filename": "unit-fixture",
        "source_id": "source-a",
        "partition": "test",
        "accident_time": 6,
    }
    return run, annotation, times


def test_labels_use_future_onset_but_selection_never_uses_ground_truth_region(tmp_path):
    run, annotation, times = cached_scene(tmp_path)
    rows = collision_snapshots(run, annotation, times)
    assert [r["label"] for r in rows] == [0, 1, None]
    assert rows[1]["selected_pair_ids"] == [1, 3]
    assert rows[2]["exclusion_reason"] == "at_or_after_published_collision"
    altered_region = {**annotation, "center_x": 0.99, "x1": 0, "x2": 0.01, "region": "irrelevant"}
    assert collision_snapshots(run, altered_region, times) == rows
    prefix, _, _ = cached_scene(tmp_path / "prefix", stop=7.2)
    # Future labels come from the full observed source, not future detector frames.
    assert collision_snapshots(prefix, annotation, times) == rows


def test_missing_features_are_not_negative_labels_and_source_gaps_censor_both_classes(tmp_path):
    run, annotation, times = cached_scene(tmp_path / "empty", empty=True)
    rows = collision_snapshots(run, annotation, times)
    assert rows[1]["label"] == 1 and not rows[1]["scoreable"] and rows[1]["rule_score"] is None
    times = [i / 10 for i in range(111) if not 42 <= i <= 52]
    run, annotation, times = cached_scene(tmp_path / "gap", timestamps=times)
    for onset in (6, 9):
        row = collision_snapshots(run, {**annotation, "accident_time": onset}, times)[1]
        assert row["label"] is None and not row["eligible"]
        assert row["exclusion_reason"] == "timestamp_gap_in_future_horizon"
    with pytest.raises(ValueError, match="timing"):
        collision_snapshots(run, annotation, [t + 0.01 for t in times])


def test_metrics_keep_abstained_positives_as_misses_without_inventing_true_negatives():
    frame = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "clip_id": ["a", "b", "c", "d"],
            "source_id": ["a", "b", "c", "d"],
            "published_accident_time_s": [2, 2, 6, 6],
            "prediction_time_s": [1, 1, 1, 1],
        }
    )
    scores = np.array([0.9, np.nan, np.nan, 0.8])
    result = scene_metrics(frame, scores, 0.5, True)
    assert result["coverage"] == 0.5 and result["alert_precision"] == 0.5
    assert result["operating_recall_including_abstained_positives"] == 0.5
    assert result["accepted_only_recall"] == 1
    assert result["scored_true_negatives"] == 0 and result["abstained_negative_snapshots"] == 1
    assert result["accepted_only_brier"] == pytest.approx(0.325)
    missing = scene_metrics(frame, np.full(4, np.nan), 0.5, True)
    assert missing["alert_precision"] is None and missing["accepted_only_brier"] is None
    assert missing["operating_recall_including_abstained_positives"] == 0
    intervals = bootstrap_operating_recall(frame, {"rules_fixed_0.45": (scores, 0.5)}, repeats=50)
    assert intervals["source_groups"] == 4
    assert intervals == bootstrap_operating_recall(frame, {"rules_fixed_0.45": (scores, 0.5)}, repeats=50)


def scene_table():
    rows, cohort = [], []
    for phase in ["train", "calibration", "validation", "test"]:
        for i in range(2):
            clip = f"{phase}-{i}"
            cohort.append({"filename": clip, "partition": phase})
            for t, label, score in [(1, 0, 0.1), (4, 1, 0.8)]:
                rows.append(
                    {
                        "clip_id": clip,
                        "source_id": clip,
                        "partition": phase,
                        "nominal_time_s": t,
                        "prediction_time_s": t,
                        "published_accident_time_s": 6,
                        "future_observation_complete": True,
                        "label": label,
                        "scoreable": True,
                        "units": "px",
                        "rule_score": score,
                        "cpa_distance": 100 if not label else 5,
                        "time_to_cpa_s": 2 if not label else 0.5,
                        "closing_speed": 10 if not label else 30,
                        "quality": 0.9,
                    }
                )
    return pd.DataFrame(rows), cohort


def test_source_leakage_post_event_and_wrong_targets_are_rejected():
    frame, _ = scene_table()
    validate_scene_observations(frame)
    leaked = frame.copy()
    leaked.loc[leaked.partition == "test", "source_id"] = "train-0"
    with pytest.raises(ValueError, match="crosses"):
        validate_scene_observations(leaked)
    changed = frame.copy()
    changed.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="horizon"):
        validate_scene_observations(changed)
    changed.loc[0, "prediction_time_s"] = 7
    with pytest.raises(ValueError, match="Post-event"):
        validate_scene_observations(changed)


def test_test_features_cannot_change_training_scaler_or_calibration(tmp_path):
    frame, cohort = scene_table()
    result = run_scene_benchmark(frame, cohort, tmp_path / "first")
    assert "calibrated_logistic_validation_threshold" in result["baselines"]
    assert "calibrated_logistic_fixed_0.5" in result["baselines"]
    assert result["baselines"]["training_prior_matched_perception"]["coverage"] == 1
    first = joblib.load(tmp_path / "first/scene-baseline.joblib")
    assert list(first.estimator.feature_names_in_) == FEATURES
    changed = frame.copy()
    changed.loc[changed.partition == "test", "cpa_distance"] = 10000
    run_scene_benchmark(changed, cohort, tmp_path / "perturbed")
    second = joblib.load(tmp_path / "perturbed/scene-baseline.joblib")
    assert np.array_equal(first.estimator[1].mean_, second.estimator[1].mean_)
    assert np.array_equal(first.estimator[-1].coef_, second.estimator[-1].coef_)
    assert np.array_equal(first.calibrator.model.coef_, second.calibrator.model.coef_)


def test_missing_training_positive_reports_unavailable_model_without_fake_results(tmp_path):
    frame, cohort = scene_table()
    mask = (frame.partition == "train") & (frame.label == 1)
    frame.loc[mask, "scoreable"] = False
    frame.loc[mask, FEATURES + ["rule_score", "units"]] = None
    result = run_scene_benchmark(frame, cohort, tmp_path / "missing-training")
    assert "calibrated_logistic" in result["unavailable_baselines"]
    assert "calibrated_logistic" not in result["baselines"]
    assert not (tmp_path / "missing-training/scene-baseline.joblib").exists()


def test_missing_validation_negative_does_not_invent_a_budget_threshold(tmp_path):
    frame, cohort = scene_table()
    mask = (frame.partition == "validation") & (frame.label == 0)
    frame.loc[mask, "scoreable"] = False
    frame.loc[mask, FEATURES + ["rule_score", "units"]] = None
    result = run_scene_benchmark(frame, cohort, tmp_path / "missing-val-negatives")
    assert "calibrated_logistic_fixed_0.5" in result["baselines"]
    assert "calibrated_logistic_validation_threshold" in result["unavailable_baselines"]
    assert "calibrated_logistic_validation_threshold" not in result["baselines"]
    assert (tmp_path / "missing-val-negatives/scene-baseline.joblib").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "contradictory_score",
        "string_boolean",
        "blank_source",
        "clip_mapping",
        "cohort_partition",
        "cohort_source",
    ],
)
def test_scene_artifact_mutations_fail_closed(tmp_path, mutation):
    frame, cohort = scene_table()
    if mutation == "contradictory_score":
        frame.loc[0, "scoreable"] = False
        frame.loc[0, "rule_score"] = 0.99
    elif mutation == "string_boolean":
        frame["scoreable"] = frame.scoreable.astype(str)
    elif mutation == "blank_source":
        frame.loc[0, "source_id"] = " "
    elif mutation == "clip_mapping":
        frame.loc[0, "source_id"] = "other-source"
    elif mutation == "cohort_partition":
        cohort[0]["partition"] = "test"
    else:
        cohort[0]["source_id"] = "wrong-source"
    with pytest.raises(ValueError):
        run_scene_benchmark(frame, cohort, tmp_path / mutation)
    assert not (tmp_path / mutation).exists()
