# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit device, file, and HTTP/RTSP source selection."""

from pathlib import Path
from urllib.parse import urlparse


def parse_source(value: str) -> int | str:
    if value.isdecimal():
        return int(value)
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "rtsp"}:
        if not parsed.hostname:
            raise ValueError("Stream URL must include a hostname")
        return value
    if not Path(value).is_file():
        raise ValueError("Input file does not exist; use a device index or HTTP/RTSP URL for live input")
    return str(Path(value))
