# SPDX-License-Identifier: AGPL-3.0-only
from roadsense.dashboard.evaluation import evaluation_rows, is_scene_pilot


def test_scene_metrics_keep_coverage_and_undefined_precision_explicit():
    report = {
        "evaluation_kind": "published_scene_collision_pilot",
        "baselines": {
            "rule": {
                "alert_precision": None,
                "operating_recall_including_abstained_positives": 0,
                "coverage": 0.2,
                "accepted_only_pr_auc_average_precision": 0.5,
                "accepted_only_brier": None,
                "n_snapshots": 10,
            }
        },
    }
    row = evaluation_rows(report)[0]
    assert row["Alert precision"] is None
    assert row["Snapshot coverage"] == 0.2
    assert row["AP (accepted snapshots only)"] == 0.5
    assert row["Operating recall incl. abstentions"] == 0
    assert "PR-AUC (average precision)" not in row
    assert is_scene_pilot({"split": "frozen_original_video_source_disjoint_not_location_disjoint"})


def test_existing_event_benchmark_is_not_relabelled_as_scene_validation():
    report = {
        "synthetic": True,
        "baselines": {
            "old": {
                "precision": 0.5,
                "recall": 0.25,
                "pr_auc_average_precision": 0.4,
                "brier_score": None,
                "n_events": 20,
            }
        },
    }
    assert not is_scene_pilot(report)
    row = evaluation_rows(report)[0]
    assert row["Test episodes"] == 20 and row["Brier score"] is None
    assert "Snapshot coverage" not in row
