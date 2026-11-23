import pathlib

import pytest

from chessnets.play import uci
from chessnets.play.lichess_bot import cli as lichess_bot_cli

EXAMPLES = pathlib.Path(__file__).parents[1] / "examples"


@pytest.mark.parametrize(
    ("main", "config"),
    [
        (lichess_bot_cli.main, "lichess-frozen.yaml"),
        (uci.main, "uci-frozen.yaml"),
    ],
)
def test_validate_reports_the_overridden_configuration(main, config, capsys):
    exit_code = main(
        [
            "validate",
            "--config",
            str(EXAMPLES / config),
            "--model-id",
            "override/model",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert "'override/model'" in capsys.readouterr().out


def test_lichess_bot_validate_rejects_invalid_metadata_override():
    with pytest.raises(SystemExit):
        lichess_bot_cli.main(
            [
                "validate",
                "--config",
                str(EXAMPLES / "lichess-frozen.yaml"),
                "--metadata-mode",
                "unknown",
            ]
        )
