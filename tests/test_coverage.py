# SPDX-License-Identifier: AGPL-3.0-only
import pandas as pd
import pytest

from roadsense.evaluation.coverage import ObservationCoverage
from roadsense.evaluation.validation import load_observations


def test_gap_boundary_and_truncated_end_censor_horizons():
    coverage = ObservationCoverage(0.5)
    for t in [0, 0.5, 1, 1.5, 2, 4, 4.5, 5, 5.5, 6, 6.5, 7]:
        coverage.observe(t)
    assert coverage.assess(0, 2)["future_observation_complete"]
    assert coverage.assess(1, 3)["reason"] == "timestamp_gap_in_future_horizon"
    assert coverage.assess(3, 3)["reason"] == "prediction_outside_observation"
    assert coverage.assess(4, 3)["future_observation_complete"]
    assert coverage.assess(5, 3)["reason"] == "insufficient_future_observation"
    # Prediction time on the new segment start does not inherit the preceding gap.
    assert coverage.assess(4, 1)["future_observation_complete"]
    with pytest.raises(ValueError):
        coverage.observe(7)
    with pytest.raises(ValueError):
        coverage.assess(4, float("nan"))


def test_reconnection_breaks_coverage_even_with_small_timestamp_gap():
    coverage = ObservationCoverage()
    coverage.observe(0)
    coverage.observe(0.1)
    coverage.observe(0.2, discontinuity=True)
    coverage.observe(0.3)
    assert not coverage.assess(0.1, 0.2)["future_observation_complete"]
    assert coverage.assess(0.2, 0.1)["future_observation_complete"]


def test_censored_unknown_labels_are_excluded_before_target_validation(tmp_path):
    row = dict(
        event_id="e",
        clip_id="c",
        location_id="l",
        prediction_time_s=0,
        event_time_s=1,
        label=1,
        cpa_distance=1,
        time_to_cpa_s=1,
        closing_speed=1,
        quality=0.9,
        synthetic=False,
        future_observation_complete=True,
    )
    frame = pd.DataFrame(
        [
            row,
            dict(
                row,
                event_id="censored",
                label=None,
                future_observation_complete=False,
                future_observation_reason="timestamp_gap",
            ),
        ]
    )
    path = tmp_path / "observations.csv"
    frame.to_csv(path, index=False)
    loaded = load_observations(str(path))
    assert loaded.event_id.tolist() == ["e"]
    assert loaded.attrs["observation_coverage"]["excluded_rows"] == 1
    assert loaded.attrs["observation_coverage"]["excluded_reasons"] == {"timestamp_gap": 1}
    frame.drop(columns=["future_observation_complete"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="coverage"):
        load_observations(str(path))
    frame["future_observation_complete"] = "unknown"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="explicit true/false"):
        load_observations(str(path))
