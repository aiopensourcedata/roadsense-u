# SPDX-License-Identifier: AGPL-3.0-only
"""Validated runtime settings; secrets are read only from the environment."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    camera_id: str = Field(default="camera-local", min_length=1, max_length=100)
    location_id: str = Field(default="location-local", min_length=1, max_length=100)
    detector: Literal["yolo", "synthetic"] = "yolo"
    weights: str = "data/yolo11n.pt"
    device: str = "cpu"
    confidence: float = Field(default=0.1, gt=0, le=1)
    horizon_s: float = Field(default=3, gt=0)
    proximity_px: float = Field(default=60, gt=0)
    proximity_m: float = Field(default=3, gt=0)
    min_quality: float = Field(default=0.5, ge=0, le=1)
    alert_threshold: float = Field(default=0.45, ge=0, le=1)
    cooldown_s: float = Field(default=3, gt=0)
    max_frames: int = Field(default=0, ge=0)
    save_video: bool = False
    synthetic: bool = False
    calibration: str | None = None

    @model_validator(mode="after")
    def synthetic_guard(self):
        if self.detector == "synthetic" and not self.synthetic:
            raise ValueError("Synthetic detector requires explicit synthetic=true")
        return self

    @classmethod
    def load(cls, path: str | Path):
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROADSENSE_", env_file=".env", extra="ignore", hide_input_in_errors=True
    )
    database_url: str = Field(default="sqlite:///outputs/roadsense.db", repr=False)
    api_key: str = Field(min_length=24, repr=False)
    api_url: str = "http://127.0.0.1:8000"
    retention_days: int = Field(default=30, ge=1)
    max_request_bytes: int = Field(default=65536, ge=1024, le=1048576)
