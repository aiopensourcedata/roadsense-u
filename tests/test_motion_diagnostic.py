# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from roadsense.evaluation.motion_diagnostic import diagnose_motion_quality


def test_equal_noise_speed_controls_expose_quality_dependence_without_changing_error():
    report = diagnose_motion_quality(repeats=6, seed=42)
    assert report["synthetic"] is True
    by_name = {x["scenario"]: x for x in report["scenarios"]}

    def metric(name, key):
        return by_name[name]["metrics"][key]["value"]

    assert metric("exact_stationary", "mean_track_median_quality") == pytest.approx(0.9)
    assert metric("exact_stationary", "mean_track_velocity_rmse_px_s") == pytest.approx(0, abs=1e-10)
    assert metric("noisy_stationary", "mean_track_median_quality") < metric(
        "noisy_moving", "mean_track_median_quality"
    )
    assert metric("noisy_stationary", "mean_track_velocity_rmse_px_s") == pytest.approx(
        metric("noisy_moving", "mean_track_velocity_rmse_px_s"), abs=1e-10
    )
    assert report == diagnose_motion_quality(repeats=6, seed=42)
    with pytest.raises(ValueError):
        diagnose_motion_quality(repeats=1)
