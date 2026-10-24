# SPDX-License-Identifier: AGPL-3.0-only
"""Keep the public gallery tied to successful, labeled real-browser evidence."""

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_published_screenshot_is_used_and_matches_its_capture_manifest():
    assets = ROOT / "docs/assets"
    manifest = json.loads((assets / "provenance.json").read_text())
    readme = (ROOT / "README.md").read_text()
    assert manifest["synthetic"] is True
    assert manifest["computed_alerts"] > 0
    assert manifest["checks"] and all(manifest["checks"].values())
    entries = manifest["screenshots"]
    names = {item["file"] for item in entries}
    assert len(entries) == 5 and names == {p.name for p in assets.glob("*.png")}
    for item in entries:
        assert item["synthetic"] is True
        assert f"docs/assets/{item['file']}" in readme
        payload = (assets / item["file"]).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
        width, height = struct.unpack(">II", payload[16:24])
        assert width >= 800 and height >= 600
    assert "synthetic" in readme.lower()


def test_readme_local_links_resolve_and_gallery_alts_label_synthetic_data():
    import re
    from urllib.parse import urlsplit

    text = (ROOT / "README.md").read_text()
    for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
        parsed = urlsplit(target)
        if parsed.scheme or not parsed.path:
            continue
        assert (ROOT / parsed.path).exists(), target
    images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", text)
    assert len(images) == 5
    assert all("synthetic" in alt.lower() for alt, _ in images)
