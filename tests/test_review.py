# SPDX-License-Identifier: AGPL-3.0-only
import random

import pytest

from roadsense.evaluation.coverage import ObservationCoverage
from roadsense.evaluation.review import sample_review_windows


def test_review_sample_is_deterministic_disjoint_and_gap_free():
    times = [i / 10 for i in range(2000) if not 400 < i < 600]
    excluded = [(20, 30), (100, 110)]
    state = random.getstate()
    windows = sample_review_windows(times, excluded, count=12, seed=19)
    assert random.getstate() == state
    assert windows == sample_review_windows(times, excluded, count=12, seed=19)
    coverage = ObservationCoverage()
    for t in times:
        coverage.observe(t)
    for w in windows:
        assert times[w["prediction_frame"]] == w["prediction_time_s"]
        assert coverage.assess(w["start_s"], w["end_s"] - w["start_s"])["future_observation_complete"]
        assert not any(w["start_s"] < b and w["end_s"] > a for a, b in excluded)
    assert all(a["end_s"] < b["start_s"] for a, b in zip(windows, windows[1:]))
    assert windows != sample_review_windows(times, excluded, count=12, seed=20)


def test_review_sample_refuses_insufficient_or_invalid_coverage():
    with pytest.raises(ValueError, match="Only"):
        sample_review_windows([0, 0.1, 0.2], [], count=1)
    with pytest.raises(ValueError):
        sample_review_windows([0, 0], [], count=1)
    with pytest.raises(ValueError):
        sample_review_windows([0, 1], [(3, 2)], count=1)


def test_preparation_cli_keeps_scores_and_labels_out_of_blind_page(tmp_path):
    import csv
    import hashlib
    import json
    import subprocess
    import sys

    import cv2
    import numpy as np

    source = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(300):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    timestamps = tmp_path / "times.json"
    timestamps.write_text(
        json.dumps({"frames": [{"best_effort_timestamp_time": i / 10} for i in range(300)]})
    )
    alert = {"timestamp_s": 28.0, "track_ids": [1, 2], "heuristic_score": 0.9}
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps({"synthetic": True, "alerts": {"baseline": [alert], "guarded": [alert]}})
    )
    output = tmp_path / "review"
    command = [
        sys.executable,
        "scripts/prepare_video_review.py",
        str(source),
        "--timestamps",
        str(timestamps),
        "--comparison",
        str(comparison),
        "--output",
        str(output),
        "--count",
        "2",
        "--synthetic",
    ]
    subprocess.run(command, check=True, capture_output=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["synthetic"] and len(manifest["windows"]) == 3
    assert sum(not w["future_observation_complete"] for w in manifest["windows"]) == 1
    assert (
        hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
        == (output / "manifest.sha256").read_text().strip()
    )
    sheet = list(csv.DictReader((output / "independent-review.csv").open()))
    assert len(sheet) == 3
    assert all(not r["provisional_conflict_label"] and not r["adjudicated_label"] for r in sheet)
    page = (output / "review.html").read_text()
    assert "heuristic_score" not in page and "SYNTHETIC" in page
    assert not list(output.glob("*.jpg"))  # Footage samples are explicitly opt-in.
    assert json.loads((output / "model_context.json").read_text())
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and "Output exists" in again.stderr


def test_partial_pair_coverage_is_counted_without_claiming_scene_accuracy():
    from roadsense.evaluation.review import pair_coverage_summary

    # Counts from the 12-window development audit; numeric measurements only.
    counts = [
        (78, 0),
        (78, 1),
        (15, 0),
        (105, 6),
        (136, 1),
        (91, 0),
        (120, 0),
        (276, 1),
        (153, 0),
        (45, 1),
        (136, 0),
        (91, 10),
    ]
    result = pair_coverage_summary([{"pairs": a, "scoreable_pairs": b} for a, b in counts])
    assert result["windows"] == 12 and result["windows_without_scoreable_pair"] == 6
    assert result["reported_pairs"] == 1324 and result["scoreable_pairs"] == 20
    assert result["scoreable_pair_fraction"] == pytest.approx(20 / 1324)
    assert result["whole_scene_accuracy"] is None
    with pytest.raises(ValueError):
        pair_coverage_summary([{"pairs": 1, "scoreable_pairs": 2}])
