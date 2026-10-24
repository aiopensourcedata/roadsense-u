# SPDX-License-Identifier: AGPL-3.0-only
"""Copy reviewed source files into a fresh local Git index for publisher review."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from check_release import source_files


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/public-source-0.1.0"))
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise SystemExit("Snapshot target already exists; choose a new directory")
    if shutil.which("git") is None:
        raise SystemExit("Install Git before preparing a public-source index")
    subprocess.run([sys.executable, str(root / "scripts/check_release.py")], cwd=root, check=True)
    files = source_files(root) + [root / "data/.gitkeep", root / "outputs/.gitkeep"]
    target.mkdir(parents=True)
    for path in files:
        destination = target / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "add", "--all"], check=True)
    subprocess.run(
        [sys.executable, str(target / "scripts/check_release.py"), "--require-git"], cwd=target, check=True
    )
    print(f"Prepared {target}: reviewed source staged in a fresh local Git index.")
    print("No commit or remote publication performed. Original repository history remains unaudited.")


if __name__ == "__main__":
    main()
