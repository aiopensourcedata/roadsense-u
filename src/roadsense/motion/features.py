# SPDX-License-Identifier: AGPL-3.0-only
"""Constant-velocity interaction geometry. Undefined quantities are None, never sentinel TTCs."""

from dataclasses import asdict, dataclass

import numpy as np

from roadsense.motion.trajectories import MotionState


@dataclass
class PairFeatures:
    distance: float
    relative_speed: float
    closing_speed: float
    cpa_distance: float
    time_to_cpa_s: float
    ttc_s: float | None
    shared_time_a_s: float | None
    shared_time_b_s: float | None
    arrival_difference_s: float | None
    quality: float
    measurement_uncertainty: float
    braking_a: float
    braking_b: float
    direction_change_a_deg: float
    direction_change_b_deg: float
    units: str

    def dict(self):
        return asdict(self)


def disk_ttc(r: np.ndarray, v: np.ndarray, radius: float) -> float | None:
    """First contact of two constant-velocity disks with a known combined radius."""
    r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
    if r.shape != (2,) or v.shape != (2,) or not np.isfinite([*r, *v, radius]).all():
        raise ValueError("TTC requires finite 2D position, velocity and radius")
    if radius <= 0:
        raise ValueError("Combined radius must be positive")
    c, a = float(r @ r - radius**2), float(v @ v)
    if c <= 0:
        return 0.0
    b = 2 * float(r @ v)
    discriminant = b * b - 4 * a * c
    if a < 1e-10 or b >= 0 or discriminant < 0:
        return None
    return max(0.0, float((-b - np.sqrt(discriminant)) / (2 * a)))


def post_encroachment_time(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    """Observed occupancy intervals of the SAME defined conflict area; overlap has no PET.

    Caller must establish area geometry and observe both exits. Predicted arrivals are not PET.
    """
    if not np.isfinite([*a, *b]).all() or a[0] > a[1] or b[0] > b[1]:
        raise ValueError("Invalid observed occupancy interval")
    if a[1] <= b[0]:
        return b[0] - a[1]
    if b[1] <= a[0]:
        return a[0] - b[1]
    return None


def pair_features(
    a: MotionState,
    b: MotionState,
    horizon_s: float = 3,
    combined_radius_m: float | None = None,
    calibration_uncertainty: float = 0,
) -> PairFeatures:
    if (
        not np.isfinite([horizon_s, calibration_uncertainty]).all()
        or horizon_s <= 0
        or calibration_uncertainty < 0
    ):
        raise ValueError("Horizon must be positive and calibration uncertainty nonnegative and finite")
    for state in (a, b):
        if (
            state.position.shape != (2,)
            or state.velocity.shape != (2,)
            or not np.isfinite(
                [
                    state.timestamp,
                    *state.position,
                    *state.velocity,
                    state.quality,
                    state.residual,
                    state.speed_change,
                    state.direction_change_deg,
                ]
            ).all()
        ):
            raise ValueError("Pair measurements must be finite 2D motion states")
    if a.units not in {"px", "m"} or a.units != b.units or abs(a.timestamp - b.timestamp) > 1e-6:
        raise ValueError("Pair measurements must share coordinates and timestamp")
    r, v = b.position - a.position, b.velocity - a.velocity
    distance, speed = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    closing = -float(r @ v) / distance if distance > 1e-9 else 0.0
    tcpa = float(np.clip(-float(r @ v) / (speed**2), 0, horizon_s)) if speed > 1e-6 else 0.0
    cpa = float(np.linalg.norm(r + v * tcpa))
    ta = tb = delta = None
    matrix = np.column_stack([a.velocity, -b.velocity])
    if np.linalg.norm(a.velocity) * np.linalg.norm(b.velocity) > 1e-6 and np.linalg.cond(matrix) < 100:
        times = np.linalg.solve(matrix, r)
        if np.all(times >= 0) and np.all(times <= horizon_s):
            ta, tb = map(float, times)
            delta = abs(ta - tb)
    ttc = None
    if a.units == "m" and combined_radius_m is not None and min(a.quality, b.quality) >= 0.5:
        ttc = disk_ttc(r, v, combined_radius_m)
        if ttc is not None and ttc > horizon_s:
            ttc = None
    return PairFeatures(
        distance,
        speed,
        closing,
        cpa,
        tcpa,
        ttc,
        ta,
        tb,
        delta,
        min(a.quality, b.quality),
        float(a.residual + b.residual + 2 * calibration_uncertainty),
        a.speed_change,
        b.speed_change,
        a.direction_change_deg,
        b.direction_change_deg,
        a.units,
    )
