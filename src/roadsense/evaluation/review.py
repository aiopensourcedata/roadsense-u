# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic, outcome-independent selection of continuous review windows."""

import math
import random
from bisect import bisect_left

from roadsense.evaluation.coverage import ObservationCoverage


def sample_review_windows(
    timestamps: list[float],
    excluded: list[tuple[float, float]],
    count: int = 12,
    seed: int = 42,
    context_s: float = 2,
    horizon_s: float = 3,
    max_gap_s: float = 0.5,
) -> list[dict]:
    """Select disjoint windows without model scores or outcome labels.

    A small tail margin preserves the horizon despite frame sampling. Selection is
    a development screening sample, not a representative prevalence estimate.
    """
    if count < 1 or not all(math.isfinite(v) and v > 0 for v in (context_s, horizon_s)):
        raise ValueError("Positive count, context and horizon required")
    if any(not all(math.isfinite(x) for x in pair) or pair[1] < pair[0] for pair in excluded):
        raise ValueError("Exclusions require finite ordered intervals")
    coverage = ObservationCoverage(max_gap_s)
    for t in timestamps:
        coverage.observe(t)
    candidates = []
    duration = context_s + horizon_s + 0.1
    for start, end in coverage.segments:
        cursor = start + 0.1
        while cursor + duration <= end:
            index = bisect_left(timestamps, cursor + context_s)
            prediction = timestamps[index]
            a, b = prediction - context_s, prediction + horizon_s + 0.1
            if b <= end and not any(a < right and b > left for left, right in excluded):
                candidates.append(
                    {
                        "start_s": a,
                        "prediction_time_s": prediction,
                        "end_s": b,
                        "prediction_frame": index,
                        "future_observation_complete": True,
                    }
                )
            cursor += duration + 0.25
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} disjoint eligible windows; requested {count}")
    return sorted(random.Random(seed).sample(candidates, count), key=lambda x: x["prediction_time_s"])


def pair_coverage_summary(windows: list[dict]) -> dict:
    """Describe usable reported-pair coverage without inventing scene accuracy."""
    if not windows:
        raise ValueError("At least one prediction window is required")
    for row in windows:
        if not isinstance(row["pairs"], int) or not isinstance(row["scoreable_pairs"], int):
            raise ValueError("Pair counts must be integers")
        if not 0 <= row["scoreable_pairs"] <= row["pairs"]:
            raise ValueError("Scoreable pairs must lie between zero and reported pairs")
    total = sum(w["pairs"] for w in windows)
    accepted = sum(w["scoreable_pairs"] for w in windows)
    return {
        "windows": len(windows),
        "windows_without_scoreable_pair": sum(w["scoreable_pairs"] == 0 for w in windows),
        "windows_without_reported_pairs": sum(w["pairs"] == 0 for w in windows),
        "reported_pairs": total,
        "scoreable_pairs": accepted,
        "scoreable_pair_fraction": accepted / total if total else None,
        "whole_scene_accuracy": None,
        "limitation": "Reported-pair coverage does not establish visibility, detection recall or whole-scene negative labels.",
    }
