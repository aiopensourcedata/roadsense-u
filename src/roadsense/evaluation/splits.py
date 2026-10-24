# SPDX-License-Identifier: AGPL-3.0-only
"""Leakage checks and reproducible event/clip or location partitions."""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def validate_partitions(frame, partitions: dict[str, np.ndarray]):
    names = list(partitions)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for key in ("event_id", "clip_id"):
                if set(frame.iloc[partitions[a]][key]) & set(frame.iloc[partitions[b]][key]):
                    raise ValueError(f"{key} leakage between {a} and {b}")


def split_data(frame, mode: str = "location", seed: int = 42) -> dict[str, np.ndarray]:
    """Four disjoint sets: fitting, calibration, threshold selection, final testing.

    Duplicate clips must share canonical clip/event IDs before splitting. Hash deduplication is a data curation step.
    """
    for key in ("event_id", "clip_id", "location_id"):
        if key not in frame or frame[key].isna().any():
            raise ValueError(f"Missing {key}")
    if mode not in {"location", "event"}:
        raise ValueError("Mode must be location or event")
    if mode == "location":
        groups = frame.location_id.astype(str).to_numpy()
    else:
        # Connected components join events spanning clips and clips containing multiple event IDs.
        parent = {}

        def root(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = root(parent[x])
            return parent[x]

        for event, clip in zip(frame.event_id, frame.clip_id):
            a, b = root("e:" + str(event)), root("c:" + str(clip))
            parent[a] = b
        groups = np.array([root("e:" + str(e)) for e in frame.event_id])
    if len(set(groups)) < 4:
        raise ValueError(
            "At least four independent groups required for fitting/calibration/validation/testing"
        )
    remaining = np.arange(len(frame))
    result = {}
    for name, size in [("test", 0.25), ("validation", 1 / 6), ("calibration", 0.2)]:
        available = groups[remaining]
        if len(set(available)) < 2:
            raise ValueError("Too few groups after splitting; provide more independent events/locations")
        fit, hold = next(
            GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed).split(
                remaining, groups=available
            )
        )
        result[name], remaining = remaining[hold], remaining[fit]
    result["train"] = remaining
    validate_partitions(frame, result)
    return result
