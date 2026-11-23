from dataclasses import dataclass

import pytest

from chessnets.utils.time_control import (
    append_basic_time_info_to_prompt,
    parse_time_control,
    TimeInfoPromptStyle,
)


def test_parse_time_control_categories():
    assert parse_time_control("179+0").time_control == "Bullet"
    assert parse_time_control("180+0").time_control == "Blitz"
    assert parse_time_control("300+3").time_control == "Blitz"
    assert parse_time_control("600+0").time_control == "Rapid"
    assert parse_time_control("3600+0").time_control == "Classical"


def test_parse_time_control_missing_values():
    assert parse_time_control(None) is None
    assert parse_time_control("?") is None
    assert parse_time_control("60").time_increment == 0


def test_parse_time_control_rejects_invalid_values():
    with pytest.raises(ValueError):
        parse_time_control("300+abc")
    with pytest.raises(ValueError):
        parse_time_control("-1+0")


@dataclass
class FakeTimeInfo:
    active_player_last_move_time_spent: float = 2
    opponent_last_move_time_spent: float = 4
    active_player_time_remaining: float = 100
    opponent_time_remaining: float = 200


class FakeHistory:
    def get_time_info(self, index):
        return FakeTimeInfo()


def test_append_basic_time_info_to_prompt():
    prompt = append_basic_time_info_to_prompt(
        FakeHistory(),
        0,
        parse_time_control("300+3"),
        "Choose a move.",
        TimeInfoPromptStyle.LEGACY,
    )
    assert prompt == (
        "Choose a move. The time control for this blitz chess game was 5 minutes "
        "with 3 seconds of increment. Time taken last turn by active player: "
        "2 seconds. Time taken last turn by opponent: 4 seconds. Time remaining "
        "for the active player: 1 minute and 40 seconds. Time remaining for the "
        "opponent: 3 minutes and 20 seconds."
    )


def test_rewritten_time_info_prompt_is_the_general_default():
    prompt = append_basic_time_info_to_prompt(
        FakeHistory(),
        0,
        parse_time_control("300+3"),
        "Choose a move.",
    )

    assert prompt.endswith(
        "The time spent by the active player on their last move was 2 seconds, "
        "and the opponent spent 4 seconds. The active player had 1 minute and "
        "40 seconds remaining, while the opponent had 3 minutes and 20 seconds "
        "remaining."
    )
