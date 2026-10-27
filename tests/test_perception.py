# SPDX-License-Identifier: AGPL-3.0-only
import csv
import json

from roadsense.evaluation.perception import perception_summary


def test_perception_counts_epochs_and_does_not_infer_actor_accuracy(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"frames": 3, "config": {"min_quality": 0.5}}))
    with (tmp_path / "trajectories.csv").open("w") as stream:
        fields = ["frame", "tracking_epoch", "track_id", "class", "confidence", "quality", "flags"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch, confidence, quality, flags in [(0, 0.8, 0.6, ""), (1, 0.8, 0, "insufficient_history")]:
            writer.writerow(dict(zip(fields, [epoch, epoch, 1, "car", confidence, quality, flags])))
    (tmp_path / "features.jsonl").write_text(
        json.dumps({"abstained": True, "reasons": ["low_tracking_quality"]})
        + "\n"
        + json.dumps({"abstained": False, "reasons": []})
        + "\n"
    )
    result = perception_summary(tmp_path)
    assert result["track_ids_including_epochs"] == 2
    assert result["observations_confidence_at_least_quality_gate"] == 2
    assert result["observations_quality_at_least_gate"] == 1
    assert result["frames_with_tracks"] == 2
    assert result["scoreable_pair_observations"] == 1
    assert result["abstention_reason_counts"] == {"low_tracking_quality": 1}
    assert result["actor_accuracy"] is None
