import os
import pathlib
import subprocess
import sys

import pytest

from chessnets import cli

EXAMPLES = pathlib.Path(__file__).parents[1] / "examples"
SRC = pathlib.Path(__file__).parents[1] / "src"


def test_help_lists_top_level_commands(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("uci", "lichess-bot", "convert"):
        assert f"  {name}" in out


def test_group_help_lists_its_commands(capsys):
    assert cli.main(["convert", "--help"]) == 0
    assert "  lc0" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [[], ["convert"]])
def test_missing_command_prints_usage_and_fails(argv, capsys):
    assert cli.main(argv) == 2
    assert "usage:" in capsys.readouterr().err


def test_unknown_command_fails(capsys):
    assert cli.main(["convert", "maia2"]) == 2
    assert "unknown command 'maia2'" in capsys.readouterr().err


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("chessnets ")


def test_subcommand_receives_remaining_arguments(capsys):
    argv = ["uci", "validate", "--config", str(EXAMPLES / "uci-frozen.yaml")]
    assert cli.main(argv) == 0
    assert "Configuration is valid" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "usage"),
    [
        (["uci", "--help"], "usage: chessnets uci"),
        (["lichess-bot", "--help"], "usage: chessnets lichess-bot"),
        (["convert", "lc0", "--help"], "usage: chessnets convert lc0"),
    ],
)
def test_subcommand_help_uses_the_nested_program_name(argv, usage, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith(usage)


def test_convert_lc0_without_adapter_exits_with_explanation(monkeypatch, tmp_path):
    from chessnets.loaders import load_lc0_model

    def missing_adapter():
        raise ModuleNotFoundError("requires the 'chessnets-lc0-protobuf' adapter")

    monkeypatch.setattr(load_lc0_model, "_get_protobuf_adapter", missing_adapter)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["convert", "lc0", str(tmp_path / "in"), str(tmp_path / "out")])
    assert "chessnets-lc0-protobuf" in str(excinfo.value.code)


def test_help_does_not_import_torch():
    env = dict(
        os.environ, PYTHONPATH=str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    script = (
        "import sys; from chessnets import cli; "
        "assert cli.main(['--help']) == 0; "
        "import chessnets.play.uci; "
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "False"
