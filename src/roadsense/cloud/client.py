# SPDX-License-Identifier: AGPL-3.0-only
"""Metadata-only HTTP transport with bounded retry, timeouts and idempotent UUIDs."""

import time
from urllib.parse import urlparse

import httpx

from roadsense.cloud.schemas import TrafficEvent


class EventClient:
    def __init__(self, base_url: str, api_key: str, attempts: int = 3, transport=None):
        if not 1 <= attempts <= 10:
            raise ValueError("Request attempts must be between 1 and 10")
        parsed = urlparse(base_url)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Keep credentials out of the API URL; use the API key header")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("API URL must be HTTP(S)")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1", "api"}:
            raise ValueError("Remote API connections require HTTPS")
        self.http = httpx.Client(
            trust_env=False,
            base_url=base_url,
            headers={"X-API-Key": api_key},
            timeout=5,
            transport=transport,
            follow_redirects=False,
        )
        self.attempts, self.bytes_attempted, self.events_sent = attempts, 0, 0

    def submit(self, event: TrafficEvent):
        body = event.model_dump_json().encode()
        for attempt in range(self.attempts):
            try:
                self.bytes_attempted += len(body)
                response = self.http.post(
                    "/events", content=body, headers={"Content-Type": "application/json"}
                )
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()
                    self.events_sent += 1
                    return response.json()
                response.raise_for_status()
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code < 500
                    and exc.response.status_code != 429
                ):
                    raise
                if attempt + 1 == self.attempts:
                    raise
                time.sleep(0.2 * 2**attempt)
        raise RuntimeError("No request attempts configured")

    def close(self):
        self.http.close()
