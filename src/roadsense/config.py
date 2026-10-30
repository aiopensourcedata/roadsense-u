# SPDX-License-Identifier: AGPL-3.0-only
"""Validated runtime settings; secrets are read only from the environment."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from roadsense.cloud.auth import ServiceCredential


class PairIdentityConfig(BaseModel):
    """Conservative ambiguity thresholds, separate from geometric risk thresholds."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    min_iou: float = Field(default=0.5, gt=0, le=1)
    max_center_separation: float = Field(default=0.2, gt=0)
    window_s: float = Field(default=0.5, gt=0)
    min_span_s: float = Field(default=0.2, gt=0)
    min_samples: int = Field(default=4, ge=3)
    min_overlap_fraction: float = Field(default=0.8, gt=0, le=1)
    max_offset_drift_per_s: float = Field(default=0.25, gt=0)
    max_offset_residual: float = Field(default=0.05, gt=0)
    max_gap_s: float = Field(default=0.2, gt=0)

    @model_validator(mode="after")
    def consistent_window(self):
        if self.min_span_s > self.window_s or self.max_gap_s > self.window_s:
            raise ValueError("Identity span and gap cannot exceed its history window")
        return self


class TilingConfig(BaseModel):
    """Opt-in crop inference; grid one is a tracker-bridge regression control."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    grid: Literal[1, 2] = 2
    overlap: float = Field(default=0.2, ge=0, lt=0.5)
    nms_iou: float = Field(default=0.6, gt=0, le=1)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    camera_id: str = Field(default="camera-local", min_length=1, max_length=100)
    location_id: str = Field(default="location-local", min_length=1, max_length=100)
    detector: Literal["yolo", "synthetic"] = "yolo"
    weights: str = "data/yolo11n.pt"
    device: str = "cpu"
    confidence: float = Field(default=0.1, gt=0, le=1)
    inference_size: int = Field(default=640, ge=320, le=1920, multiple_of=32)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    horizon_s: float = Field(default=3, gt=0)
    proximity_px: float = Field(default=60, gt=0)
    proximity_m: float = Field(default=3, gt=0)
    min_quality: float = Field(default=0.5, ge=0, le=1)
    alert_threshold: float = Field(default=0.45, ge=0, le=1)
    cooldown_s: float = Field(default=3, gt=0)
    start_frame: int = Field(default=0, ge=0)
    max_frames: int = Field(default=0, ge=0)
    save_video: bool = False
    synthetic: bool = False
    calibration: str | None = None
    camera_pose_version: str = Field(default="unspecified", min_length=1, max_length=100)
    calibration_fallback: Literal["exclude", "image_space"] = "exclude"
    outbox_capacity: int = Field(default=1000, ge=1, le=100000)
    delivery_shutdown_s: float = Field(default=1.0, ge=0, le=30)
    pair_identity: PairIdentityConfig = Field(default_factory=PairIdentityConfig)
    observation_max_gap_s: float = Field(default=0.5, gt=0)

    @model_validator(mode="after")
    def synthetic_guard(self):
        if self.detector == "synthetic" and not self.synthetic:
            raise ValueError("Synthetic detector requires explicit synthetic=true")
        if self.tiling.enabled and self.detector != "yolo":
            raise ValueError("Tiled inference requires the real YOLO detector")
        return self

    @classmethod
    def load(cls, path: str | Path):
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROADSENSE_", env_file=".env", extra="ignore", hide_input_in_errors=True
    )
    database_url: str = Field(default="sqlite:///outputs/roadsense.db", repr=False)
    api_key: str = Field(default="", repr=False)
    auth_mode: Literal["development_shared_key", "scoped"] = "development_shared_key"
    service_credentials: list[ServiceCredential] = Field(default_factory=list, repr=False)
    auto_migrate: bool = True
    retention_interval_s: float = Field(default=3600, ge=0, le=86400)

    @model_validator(mode="after")
    def auth_configuration(self):
        if self.auth_mode == "development_shared_key" and len(self.api_key) < 24:
            raise ValueError("Development shared key requires at least 24 characters")
        if self.auth_mode == "scoped":
            if not self.service_credentials or len(
                {r.credential_id for r in self.service_credentials}
            ) != len(self.service_credentials):
                raise ValueError("Scoped authentication requires unique service credential IDs")
            if len({r.token_sha256 for r in self.service_credentials}) != len(self.service_credentials):
                raise ValueError("Service credential hashes must be unique")
        return self

    api_url: str = "http://127.0.0.1:8000"
    api_ca_file: str | None = None
    retention_days: int = Field(default=30, ge=1)
    max_request_bytes: int = Field(default=65536, ge=1024, le=1048576)
