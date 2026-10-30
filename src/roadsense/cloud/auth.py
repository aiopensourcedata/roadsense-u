# SPDX-License-Identifier: AGPL-3.0-only
"""Single-organization service credentials with explicit camera and role scopes."""

import hashlib
import hmac
from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServiceCredential(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credential_id: str = Field(min_length=1, max_length=100)
    token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", repr=False)
    role: Literal["ingest", "read", "admin"]
    cameras: list[str] = Field(min_length=1, max_length=1000)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_scope(self):
        if len(set(self.cameras)) != len(self.cameras) or any(
            not c.strip() or len(c) > 100 for c in self.cameras
        ):
            raise ValueError("Unique nonempty camera scopes required")
        if "*" in self.cameras and (self.role != "admin" or self.cameras != ["*"]):
            raise ValueError("Only administrators can have wildcard camera scope")
        return self


@dataclass(frozen=True)
class Principal:
    credential_id: str
    role: str
    cameras: tuple[str, ...]

    def permits_camera(self, camera: str) -> bool:
        return "*" in self.cameras or camera in self.cameras

    def require_camera(self, camera: str) -> None:
        if not self.permits_camera(camera):
            raise HTTPException(403, "Camera outside credential scope")

    def require_role(self, role: str) -> None:
        if self.role != "admin" and self.role != role:
            raise HTTPException(403, "Credential role does not permit this action")


def resolve_principal(settings, token: str) -> Principal:
    if settings.auth_mode == "development_shared_key":
        if hmac.compare_digest(token.encode(), settings.api_key.encode()):
            return Principal("development-key", "admin", ("*",))
    else:
        digest = hashlib.sha256(token.encode()).hexdigest()
        matched = None
        # Compare every configured hash, without logging raw tokens or digests.
        for record in settings.service_credentials:
            if hmac.compare_digest(digest, record.token_sha256) and record.enabled:
                matched = Principal(record.credential_id, record.role, tuple(record.cameras))
        if matched is not None:
            return matched
    raise HTTPException(401, "Invalid or disabled credential")
