# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded metadata-only API contract; no images, identity fields or arbitrary payloads."""

from typing import Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FeatureSummary(StrictModel):
    distance: float = Field(ge=0)
    relative_speed: float = Field(ge=0)
    closing_speed: float
    cpa_distance: float = Field(ge=0)
    time_to_cpa_s: float = Field(ge=0)
    ttc_s: float | None = Field(default=None, ge=0)
    shared_time_a_s: float | None = Field(default=None, ge=0)
    shared_time_b_s: float | None = Field(default=None, ge=0)
    arrival_difference_s: float | None = Field(default=None, ge=0)
    quality: float = Field(ge=0, le=1)
    measurement_uncertainty: float = Field(ge=0)
    braking_a: float = 0
    braking_b: float = 0
    direction_change_a_deg: float = Field(default=0, ge=0, le=180)
    direction_change_b_deg: float = Field(default=0, ge=0, le=180)
    units: Literal["px", "m"]

    @model_validator(mode="after")
    def valid_ttc(self):
        if self.units != "m" and self.ttc_s is not None:
            raise ValueError("Disk TTC requires metric geometry")
        return self


class TrafficEvent(StrictModel):
    event_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    tracking_epoch: int = Field(default=0, ge=0)
    camera_id: str = Field(min_length=1, max_length=100)
    location_id: str = Field(min_length=1, max_length=100)
    timestamp: AwareDatetime
    video_time_s: float = Field(ge=0)
    event_type: Literal["interaction", "predicted_conflict"]
    risk_score: float = Field(ge=0, le=1)
    score_kind: Literal["heuristic", "calibrated_probability"] = "heuristic"
    model_version: str = Field(min_length=1, max_length=100)
    track_ids: list[int] = Field(min_length=2, max_length=2)
    classes: list[str] = Field(min_length=2, max_length=2)
    features: FeatureSummary
    explanations: list[str] = Field(max_length=10)
    synthetic: bool = False

    @model_validator(mode="after")
    def validate_lists(self):
        if self.track_ids[0] == self.track_ids[1] or min(self.track_ids) < 0:
            raise ValueError("Two distinct nonnegative track IDs required")
        if any(len(s) > 300 for s in self.explanations) or any(len(s) > 30 for s in self.classes):
            raise ValueError("Metadata strings exceed maximum lengths")
        return self


class Exposure(StrictModel):
    exposure_id: UUID = Field(default_factory=uuid4)
    camera_id: str = Field(min_length=1, max_length=100)
    location_id: str = Field(min_length=1, max_length=100)
    start: AwareDatetime
    end: AwareDatetime
    road_users: int = Field(ge=0)
    synthetic: bool = False
    method: Literal["manual_count", "validated_line_crossings"]

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Exposure end must follow start")
        return self
