# SPDX-License-Identifier: AGPL-3.0-only
import csv

import pandas as pd

from roadsense.evaluation.readiness import audit_research_inputs


def test_provisional_notes_and_censored_adjudication_cannot_become_evidence(tmp_path):
    path = tmp_path / "review.csv"
    with path.open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_id",
                "provisional_conflict_label",
                "adjudicated_label",
                "future_observation_complete",
                "reviewer_id",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                dict(
                    window_id="w1",
                    provisional_conflict_label=0,
                    adjudicated_label="",
                    future_observation_complete=True,
                    reviewer_id="first",
                ),
                dict(
                    window_id="w2",
                    provisional_conflict_label="",
                    adjudicated_label=1,
                    future_observation_complete=False,
                    reviewer_id="second",
                ),
            ]
        )
    report = audit_research_inputs([path])
    assert not report["ready_to_execute_real_benchmark_under_data_contract"]
    assert report["review_sheets"][0]["declared_adjudicated_labels"] == 0
    assert report["review_sheets"][0]["adjudicated_labels_without_complete_future"] == 1
    assert report["research_hypothesis_supported"] is None


def test_contract_readiness_does_not_certify_science_or_synthetic_validation(tmp_path):
    rows = []
    for location in range(8):
        for i in range(8):
            rows.append(
                dict(
                    event_id=f"e{location}-{i}",
                    clip_id=f"c{location}-{i}",
                    location_id=f"L{location}",
                    prediction_time_s=0,
                    event_time_s=1 if i % 2 else None,
                    label=i % 2,
                    cpa_distance=10,
                    time_to_cpa_s=1,
                    closing_speed=2,
                    quality=0.8,
                    synthetic=False,
                    future_observation_complete=True,
                )
            )
    p = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    report = audit_research_inputs([], p)
    assert report["ready_to_execute_real_benchmark_under_data_contract"]
    assert report["research_hypothesis_supported"] is None and report["novelty_established"] is None
    assert any("adjudicated" in x for x in report["blocking_evidence"])
    frame = pd.DataFrame(rows)
    frame["synthetic"] = True
    frame.to_csv(p, index=False)
    report = audit_research_inputs([], p)
    assert not report["ready_to_execute_real_benchmark_under_data_contract"]
    assert any("Synthetic" in x for x in report["blocking_evidence"])


def test_readiness_cli_fails_closed_when_no_real_evidence_exists(tmp_path):
    import json
    import subprocess
    import sys

    output = tmp_path / "audit.json"
    result = subprocess.run(
        [sys.executable, "scripts/assess_research_readiness.py", "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    report = json.loads(output.read_text())
    assert report["observations"]["provided"] is False
    assert report["research_hypothesis_supported"] is None
    assert report["blocking_evidence"]
    assert "Traceback" not in result.stderr
