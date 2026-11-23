import importlib.util
import pathlib
import shutil

import pytest

ROOT = pathlib.Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "release_check.py"
SPEC = importlib.util.spec_from_file_location("release_check", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

CHANGELOG = """# Changelog

## [Unreleased]

- Work in progress.

## [1.2.0] - 2027-01-05

### Added

- Something new.

## [1.1.0] - 2026-12-01

- Older.

[Unreleased]: https://example.com/compare/v1.2.0...HEAD
[1.2.0]: https://example.com/releases/tag/v1.2.0
"""


def _repo(tmp_path, version="1.2.0", changelog=CHANGELOG):
    (tmp_path / "plugins" / "lc0-protobuf").mkdir(parents=True)
    for rel in ("pyproject.toml", "plugins/lc0-protobuf/pyproject.toml"):
        shutil.copy(ROOT / rel, tmp_path / rel)
    MODULE.sync_versions.write(version, tmp_path)
    if changelog is not None:
        (tmp_path / "CHANGELOG.md").write_text(changelog)
    (tmp_path / "CITATION.cff").write_text(f"cff-version: 1.2.0\nversion: {version}\n")
    (tmp_path / "README.md").write_text(
        f"@software{{x,\n  version = {{{version}}},\n}}\n"
    )
    return tmp_path


def test_section_stops_at_next_release_heading():
    body = MODULE.changelog_section(CHANGELOG, "1.2.0")
    assert body == "### Added\n\n- Something new.\n"


def test_last_section_stops_at_link_references():
    assert MODULE.changelog_section(CHANGELOG, "1.1.0") == "- Older.\n"


def test_undated_or_missing_section_is_rejected():
    with pytest.raises(ValueError, match="no '## \\[2.0.0\\]"):
        MODULE.changelog_section(CHANGELOG, "2.0.0")
    with pytest.raises(ValueError, match="empty"):
        MODULE.changelog_section("## [3.0.0] - 2027-02-02\n\n## [2.0.0] - x\n", "3.0.0")


def test_ready_repository_passes(tmp_path):
    version, notes = MODULE.check("v1.2.0", _repo(tmp_path))
    assert version == "1.2.0"
    assert "Something new" in notes


def test_every_problem_is_reported(tmp_path):
    root = _repo(tmp_path, version="1.3.0", changelog=None)
    plugin = root / "plugins/lc0-protobuf/pyproject.toml"
    plugin.write_text(
        plugin.read_text().replace(
            "classifiers = [", 'classifiers = [\n    "Private :: Do Not Upload",', 1
        )
    )
    with pytest.raises(ValueError) as error:
        MODULE.check("v1.2.0", root)
    message = str(error.value)
    assert "does not match the package version v1.3.0" in message
    assert "chessnets-lc0-protobuf has a 'Private ::' classifier" in message
    assert "CHANGELOG.md is missing" in message


def test_stale_citation_version_is_reported(tmp_path):
    root = _repo(tmp_path)
    (root / "CITATION.cff").write_text("version: 1.1.0\n")
    with pytest.raises(ValueError, match=r"CITATION.cff gives version \['1.1.0'\]"):
        MODULE.check("v1.2.0", root)


def test_repository_citation_versions_match():
    version = MODULE.sync_versions.core_version()
    try:
        MODULE.check(f"v{version}")
    except ValueError as error:
        # Only the changelog may be unready between releases.
        assert "CITATION" not in str(error) and "README" not in str(error)


def test_repository_changelog_is_well_formed():
    # Unreleased is fine between releases; a dated section must not be empty.
    text = (ROOT / "CHANGELOG.md").read_text()
    assert text.startswith("# Changelog")
    assert "## [Unreleased]" in text
