# SPDX-License-Identifier: AGPL-3.0-only
"""Checkpoint changes must be explicit and bound before any experiment output/inference."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_checkpoint_override_requires_matching_digest_before_creating_run(tmp_path, digest):
    reference = tmp_path / "reference.fixture"
    candidate = tmp_path / "candidate.fixture"
    reference.write_bytes(b"unit fixture, not a model")
    candidate.write_bytes(b"different unit fixture, not a model")
    config = tmp_path / "config.yaml"
    config.write_text(f"weights: {reference}\n")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "risk_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "weights_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                "selected": [],
            }
        )
    )
    downloads = tmp_path / "downloads.json"
    downloads.write_text("[]")
    output = tmp_path / "run"
    command = [
        sys.executable,
        str(Path("scripts/compare_detector_resolution.py").resolve()),
        "--plan",
        str(plan),
        "--downloads",
        str(downloads),
        "--config",
        str(config),
        "--audits",
        str(tmp_path),
        "--weights",
        str(candidate),
        "--output",
        str(output),
    ]
    if digest is not None:
        command += ["--weights-sha256", digest]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert ("supplied together" if digest is None else "checksum mismatch") in result.stderr
    assert not output.exists()
