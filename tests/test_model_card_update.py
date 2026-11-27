import importlib.util
import json
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "update_model_card.py"
SPEC = importlib.util.spec_from_file_location("update_model_card", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
HEADING = MODULE.lc0.GENERATED_HEADING


def _repo(tmp_path, name, readme=None, config=None):
    repo = tmp_path / name
    repo.mkdir()
    if readme is not None:
        (repo / "README.md").write_text(readme, encoding="utf-8")
    if config is not None:
        (repo / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return repo


@pytest.mark.parametrize(
    ("name", "config", "expected"),
    [
        ("maia-1500", None, "maia1"),
        ("lc0_BT4-1024x15x32h", None, "lc0"),
        ("maia2-blitz", None, "maia2"),
        ("renamed", {"model_type": "maia2"}, "maia2"),
        ("renamed", {"model_type": "lc0"}, "lc0"),
    ],
)
def test_detect_family(tmp_path, name, config, expected):
    assert MODULE.detect_family(_repo(tmp_path, name, config=config)) == expected


def test_maia1_card_preserves_sources_and_is_idempotent(tmp_path):
    repo = _repo(
        tmp_path,
        "maia-1500",
        "---\nlicense: gpl-3.0\n---\n\n# Maia\n\nSource: https://example.test/model\n",
    )

    assert MODULE.update_model_card(repo, write=True, repository_id="owner/maia-1500")
    first = (repo / "README.md").read_text(encoding="utf-8")
    assert first.startswith("---\nlicense: gpl-3.0\n---\n\n# Maia\n")
    assert "Source: https://example.test/model" in first
    assert 'from_pretrained("owner/maia-1500")' in first
    assert "mcilroyyoung2020maia" in first
    assert first.count(HEADING) == 1

    assert not MODULE.update_model_card(
        repo, write=True, repository_id="owner/maia-1500"
    )
    assert (repo / "README.md").read_text(encoding="utf-8") == first


@pytest.mark.parametrize(
    ("architecture", "transformer"),
    [
        ("Lc0ChessFormerModel", True),
        ("Lc0SeResNetModel", False),
        ("BTLeelaZeroNet", True),
        ("SeLeelaZeroNet", False),
    ],
)
def test_lc0_card_cites_by_architecture(tmp_path, architecture, transformer):
    repo = _repo(
        tmp_path,
        "lc0-network",
        f"# Lc0 network\n\n{HEADING}\n\nstale\n",
        {"model_type": "lc0", "architectures": [architecture]},
    )

    MODULE.update_model_card(repo, write=True, repository_id="owner/lc0-network")
    result = (repo / "README.md").read_text(encoding="utf-8")

    assert "stale" not in result
    assert result.count(HEADING) == 1
    assert "Leela Chess Zero project" in result
    assert "mcilroyyoung2020maia" not in result
    assert ("monroe2026chessformer" in result) is transformer
    assert ("monroe2024mastering" in result) is transformer
    assert ("has no\npaper of its own" in result) is not transformer


@pytest.mark.parametrize("config", [None, {"architectures": ["Unknown"]}])
def test_lc0_card_needs_a_known_architecture(tmp_path, config):
    repo = _repo(tmp_path, "lc0-network", "# Lc0 network\n", config)
    with pytest.raises(SystemExit):
        MODULE.update_model_card(repo, repository_id="owner/lc0-network")


def test_family_override_and_crlf_normalization(tmp_path):
    repo = _repo(tmp_path, "custom", "# Custom\r\n")

    MODULE.update_model_card(repo, write=True, repository_id="o/c", family="maia1")
    result = (repo / "README.md").read_bytes()

    assert b"\r\n" not in result
    assert b"mcilroyyoung2020maia" in result


def test_duplicate_generated_heading_is_rejected(tmp_path):
    repo = _repo(tmp_path, "maia-1100", f"# M\n\n{HEADING}\n\n{HEADING}\n")
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.update_model_card(repo, repository_id="o/m")


def test_maia2_card_is_complete_and_idempotent(tmp_path):
    repo = _repo(tmp_path, "maia2-rapid", config={"time_control": "rapid"})

    assert MODULE.update_model_card(repo, write=True, repository_id="o/maia2-rapid")
    first = (repo / "README.md").read_text(encoding="utf-8")
    assert not MODULE.update_model_card(repo, write=True, repository_id="o/maia2-rapid")
    assert (repo / "README.md").read_text(encoding="utf-8") == first
    assert first.startswith("---\nlicense: mit\n")
    assert 'from_pretrained("o/maia2-rapid"' in first
    assert MODULE.maia2.CHECKPOINT_URLS["rapid"] in first
    assert "Zhenwei Tang" in first


def test_maia2_requires_time_control(tmp_path):
    repo = _repo(tmp_path, "maia2-x", config={"model_type": "maia2"})
    with pytest.raises(ValueError, match="blitz or rapid"):
        MODULE.update_model_card(repo, repository_id="o/x")


def test_preview_does_not_write(tmp_path, capsys):
    repo = _repo(tmp_path, "maia-1100", "# Maia\n")

    assert MODULE.update_model_card(repo, repository_id="o/m")
    assert (repo / "README.md").read_text(encoding="utf-8") == "# Maia\n"
    assert "(proposed)" in capsys.readouterr().out


def test_cli_rejects_repository_id_for_multiple_repos(tmp_path):
    with pytest.raises(SystemExit):
        MODULE.main(["--repository-id", "o/x", str(tmp_path), str(tmp_path)])
