# SPDX-License-Identifier: AGPL-3.0-only
"""Create local-only development credentials without displaying them."""

import secrets
from pathlib import Path

if __name__ == "__main__":
    target = Path(".env")
    if target.exists():
        raise SystemExit(".env already exists; kept existing configuration")
    key, password = secrets.token_hex(24), secrets.token_hex(20)
    with target.open("x") as f:
        target.chmod(0o600)
        f.write(
            f"ROADSENSE_API_KEY={key}\nPOSTGRES_PASSWORD={password}\n"
            f"ROADSENSE_DATABASE_URL=postgresql+psycopg://roadsense:{password}@localhost:5432/roadsense\n"
            "ROADSENSE_API_URL=http://127.0.0.1:8000\nROADSENSE_RETENTION_DAYS=30\n"
        )
    print("Created .env with private local development credentials; values not displayed.")
