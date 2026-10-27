# SPDX-License-Identifier: AGPL-3.0-only
"""Audit available research inputs without treating provisional notes as ground truth."""

import csv
import hashlib
from pathlib import Path

from roadsense.evaluation.splits import split_data
from roadsense.evaluation.validation import load_observations


def audit_research_inputs(review_paths: list[Path], observations: Path | None = None, seed: int = 42) -> dict:
    """Report mechanical data readiness; a schema cannot prove scientific validity.

    Declared reviewer IDs/adjudication are audited as declarations, not verified
    independence. Dataset-specific annotation evidence still needs human review.
    """
    reviews = []
    for path in review_paths:
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        valid = []
        censored_labels = 0
        invalid_labels = 0
        for row in rows:
            label = row.get("adjudicated_label", "").strip()
            if not label:
                continue
            if label not in {"0", "1"}:
                invalid_labels += 1
                continue
            if row.get("future_observation_complete", "").lower() not in {"true", "1"}:
                censored_labels += 1
                continue
            valid.append(int(label))
        reviews.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": len(rows),
                "provisional_labels": sum(
                    r.get("provisional_conflict_label", "") in {"0", "1"} for r in rows
                ),
                "declared_adjudicated_labels": len(valid),
                "declared_positive_labels": sum(valid),
                "declared_negative_labels": len(valid) - sum(valid),
                "invalid_adjudicated_labels": invalid_labels,
                "adjudicated_labels_without_complete_future": censored_labels,
                "distinct_declared_reviewer_ids": len(
                    {r.get("reviewer_id", "").strip() for r in rows if r.get("reviewer_id", "").strip()}
                ),
            }
        )
    blockers = []
    data = {"provided": observations is not None, "contract_ready": False}
    if observations is None:
        blockers.append("No independently labeled pair-level observation CSV was supplied.")
    else:
        data.update(file=observations.name, sha256=hashlib.sha256(observations.read_bytes()).hexdigest())
        try:
            frame = load_observations(str(observations))
            data.update(
                rows=len(frame),
                locations=int(frame.location_id.nunique()),
                class_counts={str(k): int(v) for k, v in frame.label.value_counts().items()},
                synthetic=bool(frame.synthetic.iloc[0]),
                observation_coverage=frame.attrs["observation_coverage"],
            )
            if data["synthetic"]:
                blockers.append("Synthetic observations cannot establish real-world validation.")
            if set(frame.label) != {0, 1}:
                blockers.append("Both labeled positive and negative observations are required.")
            partitions = split_data(frame, "location", seed)
            data["partitions"] = {
                name: {
                    "rows": len(indices),
                    "locations": int(frame.iloc[indices].location_id.nunique()),
                    "classes": sorted(int(x) for x in frame.iloc[indices].label.unique()),
                }
                for name, indices in partitions.items()
            }
            if any(x["classes"] != [0, 1] for x in data["partitions"].values()):
                blockers.append(
                    "A frozen train/calibration/validation/test partition lacks one target class; do not search seeds using test performance."
                )
            data["contract_ready"] = not blockers
            data["test_location_bootstrap_possible"] = data["partitions"]["test"]["locations"] >= 2
        except (ValueError, KeyError) as exc:
            data["validation_error"] = str(exc)
            blockers.append("Observation data does not meet the causal, coverage and split contract.")
    if not any(r["declared_adjudicated_labels"] for r in reviews):
        blockers.append(
            "No admissible adjudicated labels were found in the supplied review sheets; provisional labels are not ground truth."
        )
    if any(
        r["adjudicated_labels_without_complete_future"] or r["invalid_adjudicated_labels"] for r in reviews
    ):
        blockers.append("A review sheet contains invalid or censored adjudicated labels.")
    return {
        "seed": seed,
        "review_sheets": reviews,
        "observations": data,
        "ready_to_execute_real_benchmark_under_data_contract": data["contract_ready"],
        "research_hypothesis_supported": None,
        "novelty_established": None,
        "blocking_evidence": blockers,
        "limitations": [
            "A structural audit cannot verify true reviewer independence, annotation accuracy, dataset permission or source identity.",
            "Even a ready data contract is not evidence that the proposed method improves results.",
            "Actual confidence intervals, paired ablations, external-camera evaluation and declared resource/false-alarm budgets remain required.",
            "Window-screening labels cannot be broadcast to every detected pair.",
        ],
    }
