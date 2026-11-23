from chessnets.play.config import LichessBotConfig


def test_lichess_config_loads_nested_yaml_shape():
    config = LichessBotConfig.from_dict(
        {
            "model": {"id": "maia2_blitz", "device": "cpu"},
            "metadata": {
                "mode": "frozen",
                "values": {
                    "bot_elo": 1500,
                    "opponent_elo": 1300,
                    "time_control": "300+3",
                },
            },
            "challenges": {"rated": False, "max_concurrent_games": 1},
        }
    )
    assert config.model_id == "maia2_blitz"
    assert config.metadata.values["bot_elo"] == 1500
    assert config.challenges.rated is False


def test_lichess_config_rejects_unknown_fields():
    try:
        LichessBotConfig.from_dict({"model": {"id": "test"}, "modle": {}})
    except ValueError as exc:
        assert "Unknown root" in str(exc)
    else:
        raise AssertionError("unknown configuration fields must be rejected")
