# SPDX-License-Identifier: AGPL-3.0-only
import numpy as np
import pytest

from roadsense.calibration.ground_plane import GroundPlane
from roadsense.motion.features import disk_ttc, pair_features, post_encroachment_time
from roadsense.motion.trajectories import MotionState, Trajectories


def state(tid, p, v, units="px", quality=1):
    return MotionState(
        tid, 1, np.array(p, dtype=float), np.array(v, dtype=float), np.zeros(2), quality, 0, units, []
    )


def test_crossing_geometry():
    f = pair_features(state(1, [-10, 0], [5, 0]), state(2, [0, -10], [0, 5]))
    assert f.cpa_distance == pytest.approx(0)
    assert f.time_to_cpa_s == pytest.approx(2)
    assert f.shared_time_a_s == pytest.approx(2)
    assert f.arrival_difference_s == pytest.approx(0)
    assert f.ttc_s is None


def test_disk_ttc_and_divergence():
    assert disk_ttc(np.array([10.0, 0]), np.array([-2.0, 0]), 2) == pytest.approx(4)
    assert disk_ttc(np.array([10.0, 0]), np.array([2.0, 0]), 2) is None
    assert disk_ttc(np.array([10.0, 5]), np.array([-2.0, 0]), 2) is None
    assert disk_ttc(np.zeros(2), np.zeros(2), 2) == 0
    f = pair_features(state(1, [0, 0], [1, 0], "m"), state(2, [5, 0], [-1, 0], "m"), combined_radius_m=1)
    assert f.ttc_s == pytest.approx(2)


def test_pet_requires_nonoverlapping_observations():
    assert post_encroachment_time((1, 2), (4, 5)) == 2
    assert post_encroachment_time((4, 5), (1, 2)) == 2
    assert post_encroachment_time((1, 4), (3, 5)) is None
    with pytest.raises(ValueError):
        post_encroachment_time((4, 1), (3, 5))


def test_causal_velocity_and_gap():
    trajectory = Trajectories()
    for t in np.arange(0, 1, 0.1):
        s = trajectory.update(1, float(t), np.array([10 * t, 5 * t]), 0.9)
    assert s.velocity == pytest.approx([10, 5])
    assert s.quality == pytest.approx(0.9)
    s = trajectory.update(1, 3, np.array([30.0, 15]), 0.9)
    assert s.quality == 0
    assert "gap_or_coordinate_change_reset" in s.flags
    with pytest.raises(ValueError):
        trajectory.update(1, 3, np.array([30.0, 15]), 0.9)


def test_calibration_hull_and_degenerate():
    plane = GroundPlane(
        [[0, 0], [100, 0], [100, 100], [0, 100]],
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        0.1,
        [[25, 25], [75, 75]],
        [[2.5, 2.5], [7.5, 7.5]],
    )
    assert plane.project(np.array([50.0, 50])) == pytest.approx([5, 5])
    assert plane.project(np.array([150.0, 50])) is None
    with pytest.raises(ValueError):
        GroundPlane([[0, 0], [1, 0], [2, 0], [3, 0]], [[0, 0], [1, 0], [2, 0], [3, 0]], 0.1, [], [])


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -1, 0])
def test_ttc_rejects_invalid_geometry(radius):
    with pytest.raises(ValueError):
        disk_ttc(np.array([10.0, 0]), np.array([-1.0, 0]), radius)


def test_calibration_rejects_duplicate_validation_points_and_invalid_tolerance():
    args = (
        [[0, 0], [100, 0], [100, 100], [0, 100]],
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        0.1,
        [[25, 25], [25, 25]],
        [[2.5, 2.5], [2.5, 2.5]],
    )
    with pytest.raises(ValueError, match="distinct"):
        GroundPlane(*args)
    with pytest.raises(ValueError, match="tolerance"):
        GroundPlane(*args, max_error_m=float("nan"))
