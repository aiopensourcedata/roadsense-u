# SPDX-License-Identifier: AGPL-3.0-only
"""Audit an allowlisted source tree and built distributions without displaying secret values.

This is a release-content guard, not a comprehensive secret or Git-history scanner.
"""

import argparse
import re
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    "README.md",
    ".streamlit/config.toml",
    "pyproject.toml",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "MANIFEST.in",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    ".gitignore",
    ".dockerignore",
}
DOCUMENTATION_IMAGES = {
    "docs/assets/overview.png",
    "docs/assets/event-evidence.png",
    "docs/assets/location-analytics.png",
    "docs/assets/evaluation.png",
    "docs/assets/live-preview.png",
}
DIRECTORIES = {"src", "scripts", "tests", "configs", "docs", ".github"}
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json"}
PRIVATE_PARTS = {".git", ".venv", ".aws", ".codex", ".agents", "__pycache__", ".pytest_cache", ".ruff_cache"}
BINARY_SUFFIXES = {
    ".pt",
    ".onnx",
    ".joblib",
    ".pkl",
    ".pickle",
    ".db",
    ".sqlite",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"),
)


def forbidden_path(name: str, package_metadata: bool = False) -> bool:
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or ".." in parts or "\\" in name:
        return True
    if any(
        part in PRIVATE_PARTS
        or part.startswith(".venv")
        or (part.endswith(".egg-info") and not package_metadata)
        for part in parts
    ):
        return True
    if path.name.startswith(".env") and path.name != ".env.example":
        return True
    if path.suffix.lower() in BINARY_SUFFIXES or path.name.endswith((".pyc", ".pyo", ".db-wal", ".db-shm")):
        return True
    for directory in ("data", "outputs"):
        if directory in parts and path.name != ".gitkeep":
            return True
    return False


def source_files(root: Path) -> list[Path]:
    """Select human-reviewable source material; never follow symlinks."""
    candidates = [root / name for name in ROOT_FILES if (root / name).exists()]
    for directory in sorted(DIRECTORIES):
        base = root / directory
        if base.is_symlink():
            raise ValueError(f"Symlinked source directory: {directory}")
        if base.exists():
            if any(p.is_symlink() for p in base.rglob("*")):
                raise ValueError(f"Source symlink inside {directory}")
            candidates.extend(
                p
                for p in base.rglob("*")
                if p.is_file()
                and (
                    p.suffix in SUFFIXES
                    or p.name == "py.typed"
                    or p.relative_to(root).as_posix() in DOCUMENTATION_IMAGES
                )
                and "__pycache__" not in p.parts
                and not any(x.endswith(".egg-info") for x in p.parts)
            )
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source symlink or path escape: {path.relative_to(root)}")
        if forbidden_path(path.relative_to(root).as_posix()):
            raise ValueError(f"Private source path: {path.relative_to(root)}")
    return sorted(set(candidates))


def local_secrets(root: Path) -> list[bytes]:
    """Read only local credential values for comparison; never return them in reports."""
    env = root / ".env"
    if not env.is_file() or env.is_symlink():
        return []
    values = []
    for line in env.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and any(word in key.upper() for word in ("KEY", "PASSWORD", "TOKEN", "SECRET")):
            value = value.strip().strip("\"'")
            if len(value) >= 16:
                values.append(value.encode())
    return values


def check_content(name: str, data: bytes, secrets: list[bytes]) -> list[str]:
    problems = []
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        problems.append(f"Potential credential signature: {name}")
    if any(secret in data for secret in secrets):
        problems.append(f"Local credential appears in release content: {name}")
    return problems


def check_artifact(path: Path, secrets: list[bytes]) -> list[str]:
    """Inspect archives without extracting them; reject traversal, links and local artifacts."""
    problems = []
    license_found = False
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            members = [
                (x.filename, x.file_size, stat.S_ISLNK(x.external_attr >> 16), lambda x=x: archive.read(x))
                for x in archive.infolist()
                if not x.is_dir()
            ]
            for name, size, is_link, read in members:
                if is_link or forbidden_path(name, package_metadata=True) or size > 2 * 1024**2:
                    problems.append(f"Forbidden or oversized archive member: {name}")
                    continue
                license_found |= PurePosixPath(name).name == "LICENSE"
                problems.extend(check_content(name, read(), secrets))
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if (
                    not member.isfile()
                    or forbidden_path(member.name, package_metadata=True)
                    or member.size > 2 * 1024**2
                ):
                    problems.append(f"Forbidden archive member: {member.name}")
                    continue
                license_found |= PurePosixPath(member.name).name == "LICENSE"
                stream = archive.extractfile(member)
                if stream is not None:
                    problems.extend(check_content(member.name, stream.read(), secrets))
    else:
        problems.append(f"Unsupported distribution format: {path.name}")
    if not license_found:
        problems.append(f"Missing project LICENSE in {path.name}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-git", action="store_true")
    parser.add_argument("--artifacts", type=Path, nargs="*")
    args = parser.parse_args()
    root = args.root.resolve()
    problems = []
    for name in sorted(ROOT_FILES | DOCUMENTATION_IMAGES):
        if not (root / name).is_file():
            problems.append(f"Missing release file: {name}")
    secrets = local_secrets(root)
    try:
        files = source_files(root)
        for path in files:
            problems.extend(check_content(path.relative_to(root).as_posix(), path.read_bytes(), secrets))
    except ValueError as exc:
        problems.append(str(exc))
        files = []
    try:
        git = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=False)
    except FileNotFoundError:
        git = None
    if git is None or git.returncode:
        print("Git index/history unavailable: source content checked; Git history is NOT certified.")
        if args.require_git:
            problems.append("A usable Git checkout is required for the tracked-file check")
    else:
        allowed = {p.relative_to(root).as_posix() for p in files} | {"data/.gitkeep", "outputs/.gitkeep"}
        for name in git.stdout.decode().split("\0"):
            if name and forbidden_path(name):
                problems.append(f"Private/generated file is tracked by Git: {name}")
            elif name and name not in allowed:
                problems.append(f"Tracked file is outside the reviewed release allowlist: {name}")
    for artifact in args.artifacts or []:
        problems.extend(check_artifact(artifact, secrets))
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(
        f"Release content checks passed for {len(files)} source files and {len(args.artifacts or [])} distributions."
    )
    print(
        "Not checked: full Git history, copyright ownership, private-reporting setup or deployment security."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
