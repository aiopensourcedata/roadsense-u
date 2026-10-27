# SPDX-License-Identifier: AGPL-3.0-only
"""Display evaluation semantics without confusing selective scene metrics with event accuracy."""


def is_scene_pilot(report: dict) -> bool:
    """Recognize current reports and the original pilot's pre-discriminator schema."""
    return report.get("evaluation_kind") == "published_scene_collision_pilot" or report.get("split") == (
        "frozen_original_video_source_disjoint_not_location_disjoint"
    )


def evaluation_rows(report: dict) -> list[dict]:
    """Preserve missing/undefined metrics, including unfit models and no-alert precision."""
    rows = []
    for name, values in report.get("baselines", {}).items():
        if is_scene_pilot(report):
            rows.append(
                {
                    "Baseline": name,
                    "Alert precision": values.get("alert_precision"),
                    "Operating recall incl. abstentions": values.get(
                        "operating_recall_including_abstained_positives"
                    ),
                    "Snapshot coverage": values.get("coverage"),
                    "AP (accepted snapshots only)": values.get("accepted_only_pr_auc_average_precision"),
                    "Brier (accepted snapshots only)": values.get("accepted_only_brier"),
                    "Test snapshots": values.get("n_snapshots"),
                }
            )
        else:
            rows.append(
                {
                    "Baseline": name,
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "PR-AUC (average precision)": values.get("pr_auc_average_precision"),
                    "Brier score": values.get("brier_score"),
                    "Test episodes": values.get("n_events"),
                }
            )
    return rows
