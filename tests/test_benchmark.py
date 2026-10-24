# SPDX-License-Identifier: AGPL-3.0-only
import json
import subprocess
import sys

import pandas as pd
import pytest

from roadsense.evaluation.benchmark import benchmark


def test_reproducible_benchmark_and_artifact_provenance(tmp_path):
    import joblib

    subprocess.run(
        [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(tmp_path / "data")], check=True
    )
    report = benchmark(str(tmp_path / "data/benchmark.csv"), str(tmp_path / "result"))
    assert report["synthetic"]
    assert report["baselines"]["rules"]["brier_score"] is None
    assert 0 <= report["baselines"]["centralized_logistic"]["brier_score"] <= 1
    assert json.loads((tmp_path / "result/evaluation.json").read_text())["seed"] == 42
    model = joblib.load(tmp_path / "result/baseline.joblib")
    test_ids = set(report["partitions"]["test"]["event_ids"])
    assert not model.used_event_ids & test_ids
    frame = pd.read_csv(tmp_path / "data/benchmark.csv")
    frame = pd.concat([frame, frame.iloc[[0]]])
    frame.to_csv(tmp_path / "duplicates.csv", index=False)
    with pytest.raises(ValueError, match="one prespecified"):
        benchmark(str(tmp_path / "duplicates.csv"), str(tmp_path / "invalid"))


def test_data_contract_rejects_mixed_provenance_and_metric_pooling(tmp_path):
    from types import SimpleNamespace

    from roadsense.evaluation.validation import load_observations, validate_holdout

    frame = pd.DataFrame(
        [
            {
                "event_id": "001",
                "clip_id": "c1",
                "location_id": "L1",
                "prediction_time_s": 0,
                "event_time_s": 1,
                "label": 1,
                "cpa_distance": 1,
                "time_to_cpa_s": 1,
                "closing_speed": 1,
                "quality": 0.9,
                "synthetic": True,
            },
            {
                "event_id": "002",
                "clip_id": "c2",
                "location_id": "L2",
                "prediction_time_s": 0,
                "event_time_s": None,
                "label": 0,
                "cpa_distance": 100,
                "time_to_cpa_s": 1,
                "closing_speed": 1,
                "quality": 0.9,
                "synthetic": False,
            },
        ]
    )
    path = tmp_path / "input.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="synthetic and real"):
        load_observations(str(path))
    frame["synthetic"] = True
    frame["units"] = "m"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="pixel-space"):
        load_observations(str(path))
    frame["units"] = "px"
    frame.to_csv(path, index=False)
    loaded = load_observations(str(path))
    assert loaded.event_id.iloc[0] == "001"
    with pytest.raises(ValueError, match="lacks split provenance"):
        validate_holdout(SimpleNamespace(), loaded)
