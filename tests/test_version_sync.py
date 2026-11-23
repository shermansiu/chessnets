import importlib.util
import pathlib
import shutil

import pytest

ROOT = pathlib.Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "sync_versions.py"
SPEC = importlib.util.spec_from_file_location("sync_versions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _copy_projects(tmp_path):
    (tmp_path / "plugins" / "lc0-protobuf").mkdir(parents=True)
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copy(
        ROOT / "plugins" / "lc0-protobuf" / "pyproject.toml",
        tmp_path / "plugins" / "lc0-protobuf" / "pyproject.toml",
    )
    return tmp_path


def test_repository_versions_are_in_lockstep():
    assert MODULE.check() == []


def test_write_updates_every_field_and_only_those(tmp_path):
    root = _copy_projects(tmp_path)
    core_before = (root / "pyproject.toml").read_text()

    MODULE.write("1.2.3rc1", root)

    assert MODULE.check(root) == []
    assert MODULE.core_version(root) == "1.2.3rc1"
    core_after = (root / "pyproject.toml").read_text()
    changed = [
        (old, new)
        for old, new in zip(core_before.splitlines(), core_after.splitlines())
        if old != new
    ]
    assert len(changed) == 2


def test_check_reports_drift(tmp_path):
    root = _copy_projects(tmp_path)
    plugin = root / "plugins" / "lc0-protobuf" / "pyproject.toml"
    plugin.write_text(plugin.read_text().replace('version = "', 'version = "9', 1))

    assert any("version is 9" in error for error in MODULE.check(root))


@pytest.mark.parametrize(
    ("part", "expected"),
    [("major", "1.0.0"), ("minor", "0.2.0"), ("patch", "0.1.1")],
)
def test_bump(part, expected):
    assert MODULE.bump("0.1.0", part) == expected


def test_rejects_invalid_versions(tmp_path):
    with pytest.raises(SystemExit):
        MODULE.write("not-a-version", _copy_projects(tmp_path))
    with pytest.raises(SystemExit):
        MODULE.bump("0.1.0rc1", "patch")
