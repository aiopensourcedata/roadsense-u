# SPDX-License-Identifier: AGPL-3.0-only
"""Controlled motion-quality diagnostics with known synthetic trajectories."""

import numpy as np

from roadsense.motion.trajectories import Trajectories


def diagnose_motion_quality(repeats: int = 100, seed: int = 42) -> dict:
    """Compare equal-noise trajectories; intervals resample tracks, never frames.

    This diagnoses the existing production quality score without changing it.
    Synthetic measurement error is not traffic-conflict validation.
    """
    if repeats < 2:
        raise ValueError("At least two independent trajectories are required")
    scenarios = [
        ("exact_stationary", 0, 0, False, False),
        ("noisy_stationary", 0, 1, False, False),
        ("noisy_slow", 5, 1, False, False),
        ("noisy_moving", 20, 1, False, False),
        ("larger_noise", 20, 3, False, False),
        ("single_measurement_jump", 20, 1, True, False),
        ("observation_gap", 20, 1, False, True),
    ]
    results = []
    for name, speed, sigma, jump, gap in scenarios:
        values = []
        for repeat in range(repeats):
            # Matching seeds make noise identical between same-sigma speed controls.
            rng = np.random.default_rng(np.random.SeedSequence([seed, repeat]))
            times = np.arange(90) / 30
            noise = rng.normal(0, sigma, (len(times), 2))
            motion = Trajectories()
            qualities = []
            errors = []
            flags = 0
            for i, t in enumerate(times):
                if gap and 1.2 < t < 1.9:
                    continue
                measured = np.array([100 + speed * t, 100.0]) + noise[i]
                if jump and i == 45:
                    measured[0] += 80
                state = motion.update(1, float(t), measured, 0.9)
                if t >= 1:
                    qualities.append(state.quality)
                    if "insufficient_history" not in state.flags:
                        errors.append(float(np.sum((state.velocity - np.array([speed, 0.0])) ** 2)))
                    flags += bool(state.flags)
            values.append(
                [
                    float(np.median(qualities)),
                    float(np.mean(np.array(qualities) >= 0.5)),
                    float(np.sqrt(np.mean(errors))),
                    flags / len(qualities),
                ]
            )
        values = np.asarray(values)
        bootstrap_rng = np.random.default_rng(seed)
        bootstrap = np.array(
            [values[bootstrap_rng.integers(0, repeats, repeats)].mean(axis=0) for _ in range(300)]
        )
        metrics = {}
        for i, key in enumerate(
            (
                "mean_track_median_quality",
                "mean_track_scoreable_fraction",
                "mean_track_velocity_rmse_px_s",
                "mean_track_flagged_fraction",
            )
        ):
            metrics[key] = {
                "value": float(values[:, i].mean()),
                "track_bootstrap_95_interval": [
                    float(x) for x in np.quantile(bootstrap[:, i], [0.025, 0.975])
                ],
            }
        results.append(
            {
                "scenario": name,
                "true_speed_px_s": speed,
                "measurement_noise_sigma_px": sigma,
                "independent_trajectories": repeats,
                "metrics": metrics,
            }
        )
    return {
        "synthetic": True,
        "seed": seed,
        "fps": 30,
        "duration_s": 3,
        "warmup_excluded_s": 1,
        "quality_threshold": 0.5,
        "velocity_rmse_population": "States with sufficient history; flagged zero-velocity placeholders are excluded.",
        "detector_confidence": 0.9,
        "scenarios": results,
        "interpretation": "Known measurement perturbations test the existing quality heuristic. Quality depends on estimated motion magnitude, so equal absolute error can produce different scoring coverage. This alone establishes neither a faulty estimator nor better real traffic prediction.",
        "limitations": [
            "Synthetic image-space trajectories only; no detector or labeled traffic conflicts.",
            "Bootstrap units are independent generated trajectories; matching-noise scenarios are paired.",
            "No production thresholds or quality formula were tuned.",
        ],
    }
