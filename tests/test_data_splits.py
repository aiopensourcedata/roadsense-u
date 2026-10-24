# SPDX-License-Identifier: AGPL-3.0-only
import numpy as np
import pandas as pd
import pytest

from roadsense.evaluation.metrics import early_warning, metrics, select_threshold
from roadsense.evaluation.splits import split_data, validate_partitions


def fixture():
    return pd.DataFrame(
        [
            {"event_id": f"e{i // 2}", "clip_id": f"c{i // 2}", "location_id": f"L{i // 12}"}
            for i in range(120)
        ]
    )


@pytest.mark.parametrize("mode", ["location", "event"])
def test_no_leakage(mode):
    data = fixture()
    parts = split_data(data, mode)
    validate_partitions(data, parts)
    assert sum(map(len, parts.values())) == len(data)
    if mode == "location":
        assert not set(data.iloc[parts["train"]].location_id) & set(data.iloc[parts["test"]].location_id)
    assert all(np.array_equal(v, split_data(data, mode)[k]) for k, v in parts.items())


def test_detect_duplicate_clip_leakage():
    data = fixture()
    data.loc[119, "clip_id"] = "c0"
    with pytest.raises(ValueError, match="leakage"):
        validate_partitions(data, {"train": np.array([0]), "test": np.array([119])})


def test_threshold_uses_validation_negatives_and_warning():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8])
    threshold = select_threshold(y, p, 0)
    result = metrics(y, p, threshold)
    assert result["recall"] == 1 and result["false_positive_event_fraction"] == 0
    data = pd.DataFrame(
        {
            "event_id": ["a", "a", "b"],
            "label": [1, 1, 1],
            "event_time_s": [3, 3, 4],
            "prediction_time_s": [1, 2, 2],
        }
    )
    result = early_warning(data, np.array([0.9, 0.8, 0.1]), 0.5)
    assert result["median_warning_s_detected_only"] == 2
    assert result["missed_events"] == 1
