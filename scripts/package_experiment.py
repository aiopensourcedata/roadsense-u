# SPDX-License-Identifier: AGPL-3.0-only
"""Build a local, media-free cached-result package; this does not publish it."""

import argparse
import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path

from check_release import check_content, local_secrets, source_files

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a fresh package path")
    root = Path.cwd()
    # The same reviewed allowlist excludes credentials, private research and media.
    files = source_files(root)
    plan = json.loads(args.plan.read_text())
    cohort = [{k: row[k] for k in ["filename", "source_id", "partition"]} for row in plan["selected"]]
    payloads = {str(path.relative_to(root)): path.read_bytes() for path in files}
    secrets = local_secrets(root)
    for name, data in payloads.items():
        if check_content(name, data, secrets):
            raise SystemExit("Source content failed the release guard")
    for name in ["eligible-snapshots.csv", "evaluation.json"]:
        payloads["evidence/" + name] = (args.evaluation / name).read_bytes()
    payloads["evidence/cohort.json"] = (json.dumps(cohort, indent=2) + "\n").encode()
    packages = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")
    }
    payloads["evidence/environment.json"] = (json.dumps(packages, sort_keys=True, indent=2) + "\n").encode()
    payloads["REPRODUCE.md"] = b"""# Local cached-result reproduction package

This package contains current code plus historical cached measurements. The current motion guard did not produce these historical features. It is not a full acquisition/inference archive and not a release or paper submission. Review dataset-derived metadata permissions before redistribution. No media, weights, credentials, joblib models or private manuscript files are included.

With the recorded Python environment installed, from this extracted directory:

    python scripts/reproduce_scene.py --observations evidence/eligible-snapshots.csv --cohort evidence/cohort.json --expected evidence/evaluation.json --output outputs/reproduced

The command refits the saved protocol from cached features and compares the entire report, including the bootstrap. evidence/environment.json records exact local package versions without editable paths. configs/vision_environment.json and the pinned vision CI job cover the tested perception dependency boundary separately.
"""
    manifest = {
        "schema_version": 1,
        "scope": "cached-result reproduction only; current code differs from historical feature extraction",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())},
    }
    payloads["PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            archive.writestr(name, data)
    print(f"Wrote {len(payloads)} files; review redistribution rights before publishing")
