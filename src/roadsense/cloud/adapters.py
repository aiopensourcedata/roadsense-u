# SPDX-License-Identifier: AGPL-3.0-only
"""Optional future transport/storage adapter contracts; no automatic footage upload."""

from typing import Protocol

from roadsense.cloud.schemas import TrafficEvent


class EventPublisher(Protocol):
    def publish(self, event: TrafficEvent) -> None:
        """Publish compact metadata via MQTT or a managed queue with a stable event UUID."""
        ...


class ConsentedClipStore(Protocol):
    def upload(self, path: str, *, consent_reference: str, expires_at: str) -> str:
        """Explicit opt-in short clip storage with expiry and documented permission."""
        ...
