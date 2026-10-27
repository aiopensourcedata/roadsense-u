# SPDX-License-Identifier: AGPL-3.0-only
"""Regression measurements and explicit controls for ambiguity abstention."""

import json
from pathlib import Path

import numpy as np
import pytest

from roadsense.config import PairIdentityConfig
from roadsense.detection.pair_identity import PairIdentityGuard
from roadsense.detection.tracker import Detection


def detection(tid, x, cls="car"):
    return Detection((x, 0, x + 80, 40), cls, 0.9, tid)


def test_measured_bellevue_duplicate_alert_is_ambiguous_without_deleting_tracks():
    fixture = json.loads((Path(__file__).parent / "fixtures/bellevue_pair_identity.json").read_text())
    guard = PairIdentityGuard()
    result = None
    for frame in fixture["frames"]:
        guard.begin_frame(frame["timestamp_s"])
        detections = [Detection(**d) for d in frame["detections"]]
        if len(detections) != 2:
            continue
        before = [d.bbox for d in detections]
        assessment = guard.assess(*detections, frame["timestamp_s"])
        assert [d.bbox for d in detections] == before
        if frame["frame"] == fixture["baseline_alert"]["frame"]:
            result = assessment
    assert result is not None and result.ambiguous
    assert result.measurements["span_s"] >= 0.2
    assert result.measurements["overlap_fraction"] >= 0.8


def test_opposing_motion_is_not_rejected_just_for_overlap():
    guard = PairIdentityGuard()
    saw_overlap = False
    for t in np.arange(0, 1, 0.05):
        guard.begin_frame(float(t))
        assessment = guard.assess(detection(1, 100 + 50 * t), detection(2, 140 - 50 * t), float(t))
        saw_overlap |= assessment.measurements.get("iou", 0) > 0.8
        assert not assessment.ambiguous
    assert saw_overlap


def test_separate_parallel_cars_and_person_bicycle_are_preserved():
    guard = PairIdentityGuard()
    for t in np.arange(0, 1, 0.05):
        guard.begin_frame(float(t))
        assert not guard.assess(detection(1, t), detection(2, 90 + t), float(t)).ambiguous
        assert not guard.assess(detection(3, t, "person"), detection(4, t, "bicycle"), float(t)).ambiguous


def test_persistent_occlusion_is_ambiguous_not_proven_duplicate():
    # Two distinct objects can produce the same box evidence. Explicit abstention
    # trades coverage for caution; the guard must not claim to resolve this case.
    guard = PairIdentityGuard()
    for t in np.arange(0, 0.5, 0.05):
        guard.begin_frame(float(t))
        assessment = guard.assess(detection(1, t), detection(2, 5 + t), float(t))
    assert assessment.ambiguous


def test_gap_and_departure_remove_stale_identity_evidence():
    guard = PairIdentityGuard()
    for t in np.arange(0, 0.5, 0.05):
        guard.begin_frame(float(t))
        assessment = guard.assess(detection(1, t), detection(2, 5 + t), float(t))
    assert assessment.ambiguous
    guard.begin_frame(1)
    assert not guard.history
    assert not guard.assess(detection(1, 1), detection(2, 6), 1).ambiguous
    guard.clear()
    assert not guard.history
    with pytest.raises(ValueError):
        PairIdentityConfig(min_span_s=1)


def test_fluctuating_relative_motion_cannot_hide_behind_zero_average_drift():
    guard = PairIdentityGuard()
    for t in np.linspace(0, 0.5, 11):
        guard.begin_frame(float(t))
        a, b = detection(1, 100), detection(2, 100 + 10 * np.cos(8 * np.pi * t))
        if round(t * 20) % 2:
            a, b = b, a  # Pair ordering must not change the evidence.
        assessment = guard.assess(a, b, float(t))
    assert assessment.measurements["relative_offset_drift_per_s"] < 0.01
    assert assessment.measurements["relative_offset_residual"] > 0.05
    assert not assessment.ambiguous


def test_measured_separate_window_preserves_distinct_vehicle_pairs():
    fixture = json.loads((Path(__file__).parent / "fixtures/bellevue_pair_identity.json").read_text())
    guard = PairIdentityGuard()
    frames = fixture["distinct_object_control"]["frames"]
    assert len(frames) >= 20
    for frame in frames:
        guard.begin_frame(frame["timestamp_s"])
        detections = [Detection(**d) for d in frame["detections"]]
        assert not guard.assess(*detections, frame["timestamp_s"]).ambiguous
