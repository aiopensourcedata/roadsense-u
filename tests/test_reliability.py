# SPDX-License-Identifier: AGPL-3.0-only
"""Integrity, causality and coverage tradeoffs for the paired replay experiment."""

import json
import shutil

import cv2
import numpy as np
import pytest

from roadsense.config import PairIdentityConfig, PipelineConfig
from roadsense.detection.pair_identity import PairIdentityGuard
from roadsense.detection.tracker import Detection
from roadsense.evaluation.reliability import compare_reliability, policy_acceptance
from roadsense.pipeline import run


def test_pending_overlap_is_abstention_and_history_has_an_explicit_cost():
    cfg = PairIdentityConfig()
    measurements = {"iou": 0.8, "center_separation_box_diagonals": 0.05, "samples": 2}
    result = policy_acceptance(True, False, measurements, 0.15, cfg)
    assert result["geometry"] and result["identity_current"]
    assert not result["identity_pending"] and not result["identity_history_0_5"]
    # Distinct, non-overlapping objects lose early coverage under the history policy too.
    result = policy_acceptance(True, False, {}, 0.15, cfg)
    assert result["identity_pending"] and not result["identity_history_0_5"]
    mature = {**measurements, "samples": 10, "span_s": 0.5}
    assert all(policy_acceptance(True, False, mature, 0.8, cfg).values())
    assert not any(policy_acceptance(False, False, {}, 1, cfg).values())
    with pytest.raises(ValueError, match="finite"):
        policy_acceptance(True, False, {}, float("nan"), cfg)


def test_pending_policy_can_temporarily_suppress_distinct_crossing_objects():
    guard = PairIdentityGuard()
    current, pending = [], []
    for t in np.arange(0, 1.01, 0.05):
        guard.begin_frame(float(t))
        a, b = 100 + 50 * t, 140 - 50 * t
        assessment = guard.assess(
            Detection((a, 0, a + 80, 40), "car", 0.9, 1),
            Detection((b, 0, b + 80, 40), "car", 0.9, 2),
            float(t),
        )
        chosen = policy_acceptance(True, assessment.ambiguous, assessment.measurements, 1, guard.config)
        current.append(chosen["identity_current"])
        pending.append(chosen["identity_pending"])
    assert all(current)
    assert 0 < sum(pending) < len(pending)


@pytest.fixture
def reference_runs(tmp_path):
    video = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (320, 240))
    assert writer.isOpened()
    for _ in range(24):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()

    class DuplicateFixture:
        def detect(self, image, timestamp):
            x = 100 + 10 * timestamp
            delta = 0.3 - 0.5 * timestamp
            return [
                Detection((x, 20, x + 80, 60), "car", 0.9, 1),
                Detection((x + delta, 20, x + delta + 80, 60), "car", 0.9, 2),
            ]

    for name, frames in [("full", 24), ("prefix", 12), ("unguarded", 24)]:
        run(
            str(video),
            str(tmp_path / name),
            PipelineConfig(
                detector="synthetic",
                synthetic=True,
                max_frames=frames,
                pair_identity=PairIdentityConfig(enabled=name != "unguarded"),
            ),
            detector_override=DuplicateFixture(),
        )
    return tmp_path


def test_replay_reproduces_reference_and_future_cannot_change_past_decisions(reference_runs):
    root = reference_runs
    full = compare_reliability(root / "full", root / "full-comparison")
    prefix = compare_reliability(root / "prefix", root / "prefix-comparison")
    assert full["synthetic"] and full["detector_injected_for_test"]
    assert full["reference_decisions_scores_alerts_reproduced"]
    modes = full["policies"]
    assert modes["identity_current"]["emissions"] > modes["identity_pending"]["emissions"]
    assert modes["identity_pending"]["accepted"] < modes["identity_current"]["accepted"]
    for name, mode in modes.items():
        earlier = [e for e in mode["alerts"] if e["video_time_s"] <= 0.55 + 1e-9]
        expected = prefix["policies"][name]["alerts"]

        def strip_coverage(rows):
            return [{k: v for k, v in e.items() if k != "future_observation"} for e in rows]

        assert strip_coverage(earlier) == strip_coverage(expected)
    with pytest.raises(ValueError, match="already exists"):
        compare_reliability(root / "full", root / "full-comparison")


def test_mismatched_reference_fails_without_publishing_a_result(reference_runs):
    root = reference_runs
    corrupted = root / "corrupted"
    shutil.copytree(root / "full", corrupted)
    path = corrupted / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events
    events[0]["risk_score"] = 0.001
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    output = root / "invalid-result"
    with pytest.raises(ValueError, match="alert emissions"):
        compare_reliability(corrupted, output)
    assert not output.exists()


def test_unguarded_reference_and_empty_frames_are_replayed(reference_runs):
    root = reference_runs
    # Empty frames are genuine blank-image synthetic detections, not a traffic fallback.
    video = root / "synthetic.avi"
    run(str(video), str(root / "empty"), PipelineConfig(detector="synthetic", synthetic=True))
    result = compare_reliability(root / "empty", root / "empty-comparison")
    assert result["frames"] == 24 and result["reported_pair_observations"] == 0
    assert all(m["accepted_fraction"] is None for m in result["policies"].values())
    unguarded = compare_reliability(root / "unguarded", root / "unguarded-comparison")
    assert unguarded["reference_decisions_scores_alerts_reproduced"]
    assert (
        unguarded["policies"]["geometry"]["emissions"]
        > unguarded["policies"]["identity_pending"]["emissions"]
    )
    # Misdeclaring a guarded reference as unguarded cannot silently change its meaning.
    copied = root / "misdeclared"
    shutil.copytree(root / "full", copied)
    report = json.loads((copied / "report.json").read_text())
    report["config"]["pair_identity"]["enabled"] = False
    (copied / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Reference abstention"):
        compare_reliability(copied, root / "incorrect-unguarded-result")


def test_track_history_restarts_after_a_gap(reference_runs, monkeypatch):
    from roadsense.camera.video_reader import Frame

    root = reference_runs

    class Reader:
        fps = 20
        live = False

        def __init__(self, source, start_frame=0):
            self.cap = self

        def release(self):
            pass

        def __iter__(self):
            for i in range(20):
                t = i / 20 if i < 10 else 2 + (i - 10) / 20
                yield Frame(i, t, np.zeros((100, 100, 3), np.uint8), "synthetic_gap_fixture")

    class Detector:
        def detect(self, image, timestamp):
            x = 10 * timestamp
            return [
                Detection((x, 10, x + 20, 30), "car", 0.9, 1),
                Detection((100 - x, 10, 120 - x, 30), "bicycle", 0.9, 2),
            ]

    monkeypatch.setattr("roadsense.pipeline.VideoReader", Reader)
    run(
        str(root / "synthetic.avi"),
        str(root / "gaps"),
        PipelineConfig(detector="synthetic", synthetic=True),
        detector_override=Detector(),
    )
    result = compare_reliability(root / "gaps", root / "gaps-comparison")
    assert result["policies"]["identity_current"]["accepted"] > 0
    # Neither contiguous segment reaches 0.5 seconds, even though total time exceeds it.
    assert result["policies"]["identity_history_0_5"]["accepted"] == 0


def test_uniform_scale_control_is_invariant_only_with_scaled_proximity(reference_runs):
    from dataclasses import replace

    from roadsense.evaluation.scale_sensitivity import compare_scale_sensitivity, rescale_features
    from roadsense.motion.features import PairFeatures
    from roadsense.risk.rules import estimate

    feature = PairFeatures(100, 40, 40, 50, 1, None, None, None, None, 0.9, 2, -3, 4, 2, 3, "px")
    scaled = rescale_features(feature, 2)
    assert scaled.time_to_cpa_s == feature.time_to_cpa_s and scaled.quality == feature.quality
    assert scaled.braking_a == -6 and scaled.distance == 200
    assert estimate(scaled, proximity=120).score == pytest.approx(estimate(feature).score)
    assert estimate(scaled, proximity=60).score < estimate(feature).score
    with pytest.raises(ValueError, match="pixel"):
        rescale_features(replace(feature, units="m"), 2)
    for value in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            rescale_features(feature, value)
    root = reference_runs
    result = compare_scale_sensitivity(root / "full", root / "scale.json")
    assert result["synthetic"] and result["detector_injected_for_test"]
    assert result["original_decisions_reproduced"]
    for name, condition in result["conditions"].items():
        if name.endswith("normalized"):
            assert condition["acceptance_changes"] == 0
            assert condition["maximum_absolute_score_difference"] < 1e-12
            assert condition["threshold_decision_changes_among_jointly_accepted"] == 0
