# SPDX-License-Identifier: AGPL-3.0-only
"""Actor box review and class-aware matching, separate from scene collision labels."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import linear_sum_assignment

ROAD_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle", "person")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ActorBox(BaseModel):
    """Visible-extent normalized box; actor ID is local to one source clip."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    actor_id: str = Field(min_length=1, max_length=100)
    class_name: Literal["car", "motorcycle", "bus", "truck", "bicycle", "person", "unknown"]
    bbox: tuple[float, float, float, float]
    kind: Literal["physical", "depiction", "uncertain"] = "physical"
    visibility: Literal["clear", "partial", "uncertain"] = "clear"

    @model_validator(mode="after")
    def valid_box(self):
        x1, y1, x2, y2 = self.bbox
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError("Boxes must have positive extent within normalized image bounds")
        return self


class FrameReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_key: str
    reviewed: bool = False
    actors: list[ActorBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_actors(self):
        if len({a.actor_id for a in self.actors}) != len(self.actors):
            raise ValueError("Actor IDs must be unique within each frame")
        return self


class ActorReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_stage: Literal["first_pass", "adjudicated"] = "first_pass"
    reviewer_ids: list[str]
    adjudicator_id: str | None = None
    review_method: Literal["human_visual", "assistant_visual_screen", "synthetic_test"]
    frames: list[FrameReview]

    @model_validator(mode="after")
    def provenance(self):
        if not self.reviewer_ids or any(not x.strip() for x in self.reviewer_ids):
            raise ValueError("Reviewer IDs must be provided")
        if len(set(self.reviewer_ids)) != len(self.reviewer_ids):
            raise ValueError("Reviewer IDs must be distinct")
        if self.review_stage == "adjudicated" and (
            len(self.reviewer_ids) < 2
            or not self.adjudicator_id
            or not self.adjudicator_id.strip()
            or self.review_method != "human_visual"
        ):
            raise ValueError("Adjudicated labels require two human reviewers and an adjudicator ID")
        if len({f.frame_key for f in self.frames}) != len(self.frames):
            raise ValueError("Duplicate reviewed frame keys")
        return self


def box_iou(a: list | tuple, b: list | tuple) -> float:
    """IoU in a common coordinate system; validated normalized boxes in this protocol."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(actors: list[ActorBox], predictions: list[dict], iou_threshold: float = 0.5) -> dict:
    """Maximum-cardinality class-aware matching, then maximum IoU within that count.

    Uncertain regions neutralize otherwise-unmatched predictions. Depictions are
    negatives, not physical actors or ignore regions. This is sampled-frame box
    matching, not COCO AP, MOT accuracy, collision accuracy or person identity.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    valid = [
        a
        for a in actors
        if a.kind == "physical" and a.visibility != "uncertain" and a.class_name != "unknown"
    ]
    ignored = [
        a
        for a in actors
        if a.kind == "uncertain"
        or a.visibility == "uncertain"
        or (a.kind == "physical" and a.class_name == "unknown")
    ]
    for prediction in predictions:
        ActorBox.model_validate(
            {
                "actor_id": "prediction",
                "class_name": prediction["class_name"],
                "bbox": prediction["bbox"],
            }
        )
        if prediction["class_name"] not in ROAD_CLASSES:
            raise ValueError("Unsupported predicted class")
    weights = np.zeros((len(valid), len(predictions)))
    bonus = min(len(valid), len(predictions)) + 1
    for i, actor in enumerate(valid):
        for j, prediction in enumerate(predictions):
            overlap = box_iou(actor.bbox, prediction["bbox"])
            if actor.class_name == prediction["class_name"] and overlap >= iou_threshold:
                weights[i, j] = bonus + overlap
    matches = []
    if weights.size:
        ai, pi = linear_sum_assignment(-weights)
        matches = [
            {
                "actor_id": valid[i].actor_id,
                "prediction_index": int(j),
                "track_id": predictions[j].get("track_id"),
                "iou": box_iou(valid[i].bbox, predictions[j]["bbox"]),
            }
            for i, j in zip(ai, pi)
            if weights[i, j] > 0
        ]
    used = {m["prediction_index"] for m in matches}
    neutral = [
        j
        for j, p in enumerate(predictions)
        if j not in used and any(box_iou(a.bbox, p["bbox"]) >= iou_threshold for a in ignored)
    ]
    return {
        "tp": len(matches),
        "fp": len(predictions) - len(used) - len(neutral),
        "fn": len(valid) - len(matches),
        "ignored_predictions": len(neutral),
        "matches": matches,
    }


def evaluate_actor_review(bundle: Path, review_path: Path, allow_first_pass: bool = False) -> dict:
    """Fail closed on blank/partial/mismatched labels; report declared provenance."""
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    review = ActorReview.model_validate_json(review_path.read_text())
    if review.manifest_sha256 != file_sha256(manifest_path):
        raise ValueError("Review belongs to a different manifest")
    if review.review_stage != "adjudicated" and not allow_first_pass:
        raise ValueError("First-pass review requires explicit allow_first_pass; it is not adjudicated truth")
    expected = {f["frame_key"]: f for f in manifest["frames"]}
    if set(expected) != {f.frame_key for f in review.frames} or not all(f.reviewed for f in review.frames):
        raise ValueError("Every manifest frame must be explicitly reviewed, including empty frames")
    predictions_path = bundle / "predictions.json"
    predictions = json.loads(predictions_path.read_text())
    if predictions["manifest_sha256"] != review.manifest_sha256:
        raise ValueError("Predictions belong to a different manifest")
    results = {}
    for stage, frames in predictions["stages"].items():
        if set(frames) != set(expected):
            raise ValueError("Prediction frame coverage differs from review manifest")
        totals, matches = Counter(), []
        for frame in review.frames:
            measured = match_boxes(frame.actors, frames[frame.frame_key])
            totals.update({k: measured[k] for k in ("tp", "fp", "fn", "ignored_predictions")})
            matches.append({"frame_key": frame.frame_key, **measured})
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        results[stage] = {
            **dict(totals),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "frames": matches,
        }
    return {
        "evaluation_kind": "sampled_actor_boxes",
        "review_stage": review.review_stage,
        "review_method": review.review_method,
        "reviewer_ids": review.reviewer_ids,
        "adjudicator_id": review.adjudicator_id,
        "manifest_sha256": review.manifest_sha256,
        "review_sha256": file_sha256(review_path),
        "predictions_sha256": file_sha256(predictions_path),
        "iou_threshold": 0.5,
        "n_frames": len(expected),
        "stages": results,
        "limitations": [
            "Declared reviewer identities/provenance are not independently verified by software.",
            "Correlated development frames; no confidence interval or camera generalization claim.",
            "No collision participant truth, continuous MOT labels or reliable ID-switch estimate.",
        ],
    }
