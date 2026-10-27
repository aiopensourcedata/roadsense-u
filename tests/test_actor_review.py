# SPDX-License-Identifier: AGPL-3.0-only
"""Synthetic actor annotations validate tools; they are not traffic ground truth."""

import json

import pytest
from pydantic import ValidationError

from roadsense.evaluation.actor_review import (
    ActorBox,
    ActorReview,
    FrameReview,
    evaluate_actor_review,
    file_sha256,
    match_boxes,
)
from roadsense.evaluation.actor_review_page import render_actor_page


def actor(tid, box, **kwargs):
    return ActorBox(actor_id=tid, class_name="car", bbox=box, **kwargs)


def test_box_matching_prefers_two_valid_matches_over_one_higher_iou():
    truth = [actor("a", [0, 0, 0.6, 1]), actor("b", [0.19, 0, 0.79, 1])]
    predictions = [{"class_name": "car", "bbox": b} for b in [[0, 0, 0.6, 1], [0, 0, 0.4, 1]]]
    matched = match_boxes(truth, predictions)
    assert (matched["tp"], matched["fp"], matched["fn"]) == (2, 0, 0)
    assert {m["actor_id"]: m["prediction_index"] for m in matched["matches"]} == {"a": 1, "b": 0}
    predictions[0]["class_name"] = "truck"
    assert match_boxes(truth, predictions)["tp"] == 1


def test_depictions_are_negatives_and_uncertainty_is_not_a_false_positive():
    truth = [
        actor("poster", [0, 0, 0.2, 0.2], kind="depiction"),
        actor("uncertain", [0.5, 0.5, 1, 1], kind="uncertain"),
    ]
    predictions = [{"class_name": "car", "bbox": a.bbox} for a in truth]
    result = match_boxes(truth, predictions)
    assert (result["tp"], result["fp"], result["fn"], result["ignored_predictions"]) == (0, 1, 0, 1)
    assert match_boxes([], [predictions[0]])["fp"] == 1
    assert match_boxes([actor("a", [0, 0, 1, 1])], [])["fn"] == 1


def test_annotation_validation_rejects_bad_boxes_and_duplicate_actor_ids():
    for box in [[0, 0, 0, 1], [-1, 0, 1, 1], [0, 0, 1, float("nan")]]:
        with pytest.raises(ValidationError):
            actor("a", box)
    a = actor("a", [0, 0, 1, 1])
    with pytest.raises(ValidationError, match="unique"):
        FrameReview(frame_key="f", actors=[a, a])


def bundle(tmp_path):
    manifest = {"frames": [{"frame_key": "f1"}, {"frame_key": "f2"}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    digest = file_sha256(path)
    (tmp_path / "predictions.json").write_text(
        json.dumps(
            {
                "manifest_sha256": digest,
                "stages": {
                    "raw_640": {
                        "f1": [{"class_name": "car", "bbox": [0, 0, 1, 1]}],
                        "f2": [],
                    }
                },
            }
        )
    )
    review = {
        "schema_version": 1,
        "manifest_sha256": digest,
        "review_stage": "first_pass",
        "reviewer_ids": ["synthetic-fixture"],
        "review_method": "synthetic_test",
        "frames": [
            {"frame_key": "f1", "reviewed": True, "actors": [actor("a", [0, 0, 1, 1]).model_dump()]},
            {"frame_key": "f2", "reviewed": True, "actors": []},
        ],
    }
    return review


def test_evaluation_requires_complete_bound_review_and_preserves_provenance(tmp_path):
    review = bundle(tmp_path)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="First-pass"):
        evaluate_actor_review(tmp_path, path)
    result = evaluate_actor_review(tmp_path, path, allow_first_pass=True)
    assert result["review_method"] == "synthetic_test"
    assert result["review_stage"] == "first_pass"
    assert result["stages"]["raw_640"]["recall"] == 1
    review["frames"][1]["reviewed"] = False
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="Every manifest frame"):
        evaluate_actor_review(tmp_path, path, allow_first_pass=True)
    review["frames"][1]["reviewed"] = True
    review["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="different manifest"):
        evaluate_actor_review(tmp_path, path, allow_first_pass=True)


def test_adjudication_cannot_be_claimed_by_one_reviewer_or_assistant_screen(tmp_path):
    review = bundle(tmp_path)
    review.update(review_stage="adjudicated", review_method="human_visual", adjudicator_id="adjudicator")
    with pytest.raises(ValidationError, match="two human reviewers"):
        ActorReview.model_validate(review)
    review["reviewer_ids"] = ["reviewer-a", "reviewer-b"]
    assert ActorReview.model_validate(review).review_stage == "adjudicated"
    review["review_method"] = "assistant_visual_screen"
    with pytest.raises(ValidationError):
        ActorReview.model_validate(review)


def test_blind_page_has_no_predictions_or_outcomes_and_escapes_embedded_metadata():
    page = render_actor_page({"frames": [], "purpose": "</script><script>bad</script>"}, "0" * 64)
    assert "<script>bad</script>" not in page
    assert "fetch(" not in page and "localStorage" not in page
    assert "risk_score" not in page and "accident_time" not in page


def test_preparation_cli_defaults_to_metadata_and_refuses_heldout_sources(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from roadsense.config import PipelineConfig

    # Mock artifact bytes for preparation-contract tests only; no video/model inference.
    source, weights = tmp_path / "fixture.bin", tmp_path / "fixture-weights.bin"
    source.write_bytes(b"synthetic preparation fixture, not video")
    weights.write_bytes(b"synthetic preparation fixture, not model weights")
    development, audits = tmp_path / "development", tmp_path / "audits"
    run = development / "clip-00" / "640"
    run.mkdir(parents=True)
    (audits / "clip-00").mkdir(parents=True)
    config = PipelineConfig(weights=str(weights)).model_dump()
    downloads = tmp_path / "downloads.json"
    downloads.write_text(json.dumps([{"filename": "fixture.bin", "path": str(source)}]))
    plan = {
        "selected": [{"filename": "fixture.bin", "partition": "train", "width": 100, "height": 100}],
        "indices": [0],
        "sizes": [640],
        "base_config": config,
        "weights_sha256": file_sha256(weights),
        "downloads_sha256": file_sha256(downloads),
    }
    plan_path = development / "development-plan.json"
    plan_path.write_text(json.dumps(plan))
    (development / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "paired_snapshot_timestamps_identical": True,
            }
        )
    )
    (audits / "clip-00" / "source-audit.json").write_text(
        json.dumps(
            {
                "valid_binding": True,
                "source_sha256": file_sha256(source),
                "timestamps_s": [0, 0.5, 1],
            }
        )
    )
    (run / "snapshots.json").write_text(
        json.dumps(
            [
                {
                    "eligible": True,
                    "prediction_time_s": 1,
                    "nominal_time_s": 1,
                }
            ]
        )
    )
    (run / "report.json").write_text(json.dumps({"config": config, "detector_injected_for_test": False}))
    (run / "frame_timestamps.csv").write_text("frame,timestamp_s\n0,0\n1,0.5\n2,1\n")
    (run / "trajectories.csv").write_text("frame\n")
    output = tmp_path / "review"
    command = [
        sys.executable,
        str(Path("scripts/prepare_actor_review.py").resolve()),
        "--development",
        str(development),
        "--downloads",
        str(downloads),
        "--audits",
        str(audits),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert not (output / "frames").exists()
    assert all(f["image"] is None for f in json.loads((output / "manifest.json").read_text())["frames"])
    assert all(not f["reviewed"] for f in json.loads((output / "review-template.json").read_text())["frames"])
    diagnostic = json.loads((output / "diagnostic.json").read_text())
    assert diagnostic["actor_accuracy"] is None
    assert diagnostic["diagnostic"]["640"]["snapshots_with_two_or_more_raw_boxes"] is None
    (run / "report.json").write_text(
        json.dumps(
            {
                "config": {**config, "confidence": 0.9},
                "detector_injected_for_test": False,
            }
        )
    )
    command[-1] = str(tmp_path / "mismatched-confidence")
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0 and "Invalid inference provenance" in rejected.stderr
    assert not (Path(command[-1]) / "diagnostic.json").exists()
    plan["selected"][0]["partition"] = "test"
    plan_path.write_text(json.dumps(plan))
    command[-1] = str(tmp_path / "heldout-review")
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0 and "training sources only" in rejected.stderr
    assert not Path(command[-1]).exists()


def test_annotation_page_synthetic_drag_export_and_resume(tmp_path):
    pytest.importorskip("playwright")
    import base64
    import shutil

    import cv2
    import numpy as np
    from playwright.sync_api import sync_playwright

    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("Browser extra/system Chromium unavailable")
    ok, encoded = cv2.imencode(".jpg", np.zeros((300, 400, 3), dtype=np.uint8))
    assert ok
    image = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
    page_path = tmp_path / "index.html"
    page_path.write_text(
        render_actor_page(
            {
                "frames": [
                    {
                        "frame_key": "synthetic-f1",
                        "clip_index": 0,
                        "frame_index": 1,
                        "timestamp_s": 1,
                        "image": image,
                    },
                ]
            },
            "0" * 64,
        )
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 1200})
        page.goto(page_path.as_uri())
        page.wait_for_function("imageReady")
        page.locator("#reviewer").fill("synthetic-ui-test")
        page.locator("#actor").fill("test-actor")
        canvas = page.locator("#canvas").bounding_box()
        page.mouse.move(canvas["x"] + canvas["width"] * 0.1, canvas["y"] + canvas["height"] * 0.1)
        page.mouse.down()
        page.mouse.move(canvas["x"] + canvas["width"] * 0.6, canvas["y"] + canvas["height"] * 0.8)
        page.mouse.up()
        page.locator("#reviewed").check()
        with page.expect_download() as download:
            page.locator("#export").click()
        exported = tmp_path / "synthetic-ui-export.json"
        download.value.save_as(exported)
        review = json.loads(exported.read_text())
        # The human form is exercised on a synthetic image only; never use this as real truth.
        assert review["frames"][0]["reviewed"]
        assert review["frames"][0]["actors"][0]["bbox"] == pytest.approx([0.1, 0.1, 0.6, 0.8], abs=0.01)
        page.locator("#import").set_input_files(exported)
        page.wait_for_function("document.getElementById('status').textContent.includes('resumed')")
        assert page.locator("#boxes li").count() == 1
        browser.close()
