# SPDX-License-Identifier: AGPL-3.0-only
"""Create local private credential env files; never print service tokens."""

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--camera", default="camera-local")
    args = p.parse_args()
    if not args.camera.strip() or len(args.camera) > 100 or any(c in args.camera for c in "\r\n\"'"):
        p.error("Invalid camera identifier")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for role in ["admin", "read", "ingest"]:
        token = secrets.token_urlsafe(32)
        records.append(
            {
                "credential_id": f"local-{role}",
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "role": role,
                "cameras": ["*"] if role == "admin" else [args.camera],
            }
        )
        path = args.output / f"{role}.env"
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(f"ROADSENSE_API_KEY={token}\n")
    with os.fdopen(
        os.open(args.output / "server.env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as stream:
        stream.write(
            "ROADSENSE_AUTH_MODE=scoped\nROADSENSE_SERVICE_CREDENTIALS="
            + json.dumps(records, separators=(",", ":"))
            + "\n"
        )
    print("Created private server/admin/read/ingest environment files. Keep them out of version control.")
