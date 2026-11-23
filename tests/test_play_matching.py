from chessnets.play.matching import (
    ExactTimeControl,
    match_time_control,
    parse_exact_time_control,
)


def test_parse_exact_time_control_and_effective_duration():
    control = parse_exact_time_control("300+3")
    assert control == ExactTimeControl(300, 3)
    assert control.effective_seconds == 420
    assert control.category == "Blitz"


def test_match_time_control_prefers_exact_then_closest_duration():
    actual = parse_exact_time_control("300+2")
    assert match_time_control(actual, ["180+0", "300+2", "600+0"]) == actual
    assert match_time_control(actual, ["180+0", "600+0"]) == parse_exact_time_control(
        "180+0"
    )
