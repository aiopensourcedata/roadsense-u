# SPDX-License-Identifier: AGPL-3.0-only
"""Protect source/distribution exclusions and redacted secret detection."""

import io
import runpy
import tarfile
from pathlib import Path

import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_release.py"))


def test_private_paths_and_ordinary_metadata():
    forbidden = CHECK["forbidden_path"]
    for name in [
        ".env",
        ".aws/credentials",
        ".research/paper.tex",
        "project/.research/references.bib",
        "outputs/video.json",
        "data/a.avi",
        "../escape.py",
        "src/key.pem",
        ".venv-test/lib/foo.py",
        "src/a.pyc",
    ]:
        assert forbidden(name)
    assert not forbidden(".env.example")
    assert not forbidden("data/.gitkeep")
    assert not forbidden("roadsense-u-0.1/src/roadsense_u.egg-info/PKG-INFO", package_metadata=True)


def test_secret_values_are_detected_but_never_reported():
    secret = b"fake-private-development-value-for-test"
    issues = CHECK["check_content"]("src/a.py", secret, [secret])
    assert issues and secret.decode() not in " ".join(issues)
    assert CHECK["check_content"]("key", b"-----BEGIN " + b"PRIVATE KEY-----", [])


def test_source_selection_does_not_include_local_data_and_rejects_symlinks(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/module.py").write_text("print(1)")
    (tmp_path / ".env").write_text("private")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/event.json").write_text("{}")
    assert [p.relative_to(tmp_path).as_posix() for p in CHECK["source_files"](tmp_path)] == ["src/module.py"]
    (tmp_path / "src/leak.py").symlink_to(tmp_path / ".env")
    import pytest

    with pytest.raises(ValueError, match="symlink"):
        CHECK["source_files"](tmp_path)


def test_archive_inspection_rejects_private_and_traversing_members(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        for name, body in [
            ("project/LICENSE", b"license"),
            ("project/.env", b"secret"),
            ("project/.research/paper.tex", b"private manuscript fixture"),
            ("../outside.py", b"bad"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            out.addfile(info, io.BytesIO(body))
    problems = CHECK["check_artifact"](archive, [])
    assert any(".env" in p for p in problems)
    assert any(".research" in p for p in problems)
    assert any("../outside" in p for p in problems)


def test_wheel_symlinks_are_rejected(tmp_path):
    import stat
    import zipfile

    path = tmp_path / "fake.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("project.dist-info/licenses/LICENSE", "license")
        symlink = zipfile.ZipInfo("project/shortcut.py")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink, "outside.py")
    assert any("shortcut.py" in item for item in CHECK["check_artifact"](path, []))


@pytest.mark.parametrize("model", ["yolo11n", "yolo11s"])
def test_weight_download_refuses_a_changed_upstream_artifact(tmp_path, monkeypatch, model):
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/download_weights.py"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def geturl(self):
            return "https://github.com/official-test-fixture"

        def read(self, limit):
            return b"x" * (1024**2 + 1)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), "--model", model])
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(SystemExit, match="checksum differs"):
        runpy.run_path(str(script), run_name="__main__")
    assert not (tmp_path / "data" / f"{model}.pt").exists()
