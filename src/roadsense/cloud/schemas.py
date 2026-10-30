# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded metadata-only API contract; no images, identity fields or arbitrary payloads."""

from datetime import timedelta
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


class AcquisitionGeometry(StrictModel):
    width: int = Field(gt=0, le=32768)
    height: int = Field(gt=0, le=32768)
    live: bool
    image_transform: Literal["decoded_native"] = "decoded_native"
    camera_pose_version: str = Field(min_length=1, max_length=100)


class TrackerSettings(StrictModel):
    tracker_type: Literal["bytetrack"]
    track_high_thresh: float = Field(ge=0, le=1)
    track_low_thresh: float = Field(ge=0, le=1)
    new_track_thresh: float = Field(ge=0, le=1)
    track_buffer: int = Field(ge=0, le=10000)
    match_thresh: float = Field(ge=0, le=1)
    fuse_score: bool


class RunContent(StrictModel):
    schema_version: Literal[1]
    session_id: UUID
    camera_id: str = Field(min_length=1, max_length=100)
    location_id: str = Field(min_length=1, max_length=100)
    timestamp_origin: Literal["provided_capture_time", "ingestion_start_not_capture_time"]
    wall_time_origin: AwareDatetime
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    geometry: AcquisitionGeometry
    config: dict = Field(max_length=64)
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_files: dict[str, str] = Field(max_length=256)
    dependencies: dict[str, str] = Field(max_length=64)
    tracker_settings: TrackerSettings | None = None
    detector_injected_for_test: bool
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    timing: str = Field(max_length=300)

    @model_validator(mode="after")
    def validate_content(self):
        from roadsense.config import PipelineConfig
        from roadsense.provenance import canonical_digest

        PipelineConfig.model_validate(self.config)
        if canonical_digest(self.code_files) != self.code_sha256:
            raise ValueError("Code file hashes disagree with code digest")
        if (
            self.config.get("camera_id") != self.camera_id
            or self.config.get("location_id") != self.location_id
        ):
            raise ValueError("Manifest configuration disagrees with camera/location")
        return self


class RunManifest(StrictModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: RunContent

    @model_validator(mode="after")
    def verify_digest(self):
        from roadsense.provenance import canonical_digest

        if (
            len(self.model_dump_json()) > 32768
            or canonical_digest(self.content.model_dump(mode="json")) != self.sha256
        ):
            raise ValueError("Manifest digest mismatch or size limit exceeded")
        return self


class TrafficEvent(StrictModel):
    event_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    tracking_epoch: int = Field(default=0, ge=0)
    camera_id: str = Field(min_length=1, max_length=100)
    location_id: str = Field(min_length=1, max_length=100)
    timestamp: AwareDatetime
    ingested_at: AwareDatetime | None = None
    timestamp_origin: Literal[
        "provided_capture_time", "ingestion_start_not_capture_time", "legacy_unknown"
    ] = "legacy_unknown"
    timestamp_source: str = Field(default="legacy_unknown", max_length=60)
    raw_video_time_s: float | None = Field(default=None, ge=0)
    run_manifest: RunManifest | None = None
    arrival_to_alert_ms: float | None = Field(default=None, ge=0)
    capture_to_alert_ms: float | None = Field(default=None, ge=0)
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
        if self.run_manifest is not None:
            content = self.run_manifest.content
            if (
                content.session_id != self.session_id
                or content.camera_id != self.camera_id
                or content.location_id != self.location_id
                or content.timestamp_origin != self.timestamp_origin
                or abs(
                    (
                        self.timestamp - (content.wall_time_origin + timedelta(seconds=self.video_time_s))
                    ).total_seconds()
                )
                > 1e-6
            ):
                raise ValueError("Event disagrees with immutable run manifest")
        elif self.timestamp_origin != "legacy_unknown":
            raise ValueError("Explicit timing semantics require a bound run manifest")
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
