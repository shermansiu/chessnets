import pytest

from chessnets.pipelines.base import PipelineInputConfig
from chessnets.pipelines.carlini.pipeline_carlini import (
    CarliniInputMode,
    CarliniPipelineConfig,
)
from chessnets.pipelines.unimaia import UniMaiaPipelineConfig
from chessnets.utils.time_control import TimeInfoPromptStyle


def test_contract_rejects_inputs_outside_selected_mode():
    config = PipelineInputConfig.from_inputs(required=("pgns", "prompts"))

    with pytest.raises(ValueError, match="Unsupported pipeline input.*time_controls"):
        config.validate(
            pgns="1. e4 *",
            prompts="Play chess",
            time_controls="300+0",
        )


def test_contract_reports_all_missing_required_inputs():
    config = PipelineInputConfig.from_inputs(
        required=("pgns", "time_controls", "white_elos", "black_elos")
    )

    with pytest.raises(
        ValueError,
        match=r"Missing required pipeline input\(s\): black_elos, time_controls, white_elos",
    ):
        config.validate(pgns="1. e4 *")


def test_contract_checks_pair_inputs():
    config = PipelineInputConfig.from_inputs(
        required=("pgns",),
        optional=("white_clock_times", "black_clock_times"),
        paired=(("white_clock_times", "black_clock_times"),),
    )

    with pytest.raises(ValueError, match="provided together"):
        config.validate(pgns="1. e4 *", white_clock_times=[10.0])


def test_normalize_does_not_broadcast_auxiliary_values_by_default():
    config = PipelineInputConfig.from_inputs(required=("pgns", "prompts"))

    with pytest.raises(ValueError, match="expected 2"):
        config.normalize(
            primary="pgns",
            pgns=["1. e4 *", "1. d4 *"],
            prompts="Play chess",
        )


def test_normalize_can_explicitly_broadcast_an_auxiliary_value():
    config = PipelineInputConfig.from_inputs(required=("pgns", "prompts"))

    normalized = config.normalize(
        primary="pgns",
        broadcast=("prompts",),
        pgns=["1. e4 *", "1. d4 *"],
        prompts="Play chess",
    )

    assert normalized["prompts"] == ["Play chess", "Play chess"]


def test_legacy_contract_remains_permissive_about_unlisted_inputs():
    config = PipelineInputConfig(True, False, False)

    config.validate(pgns="1. e4 *", prompts="Play chess")


def test_requires_fields_use_plural_input_nouns():
    config = PipelineInputConfig(
        requires_pgns=True,
        requires_fens=False,
        requires_prompts=True,
        requires_time_controls=True,
        requires_elos=True,
        requires_elos_self_oriented=True,
    )

    assert config.required_inputs == frozenset(
        {"pgns", "prompts", "time_controls", "elos", "elo_self_oriented"}
    )


def test_aliases_cannot_be_supplied_twice():
    config = PipelineInputConfig.from_inputs(required=("pgns",))

    with pytest.raises(ValueError, match="provided more than once"):
        config.validate(pgns="1. e4 *", pgn="1. d4 *")


def test_fens_is_the_canonical_board_input():
    config = PipelineInputConfig.from_inputs(required=("fens",))

    assert config.required_inputs == frozenset({"fens"})
    assert config.allowed_inputs == frozenset({"fens"})
    config.validate(fens="startpos")

    with pytest.raises(ValueError, match="Unsupported pipeline input.*fen"):
        config.validate(fen="startpos", fens="startpos")


@pytest.mark.parametrize(
    ("mode", "required"),
    [
        (CarliniInputMode.DEFAULT, {"pgns"}),
        (CarliniInputMode.PROMPT, {"pgns", "prompts"}),
        (
            CarliniInputMode.TIME_CONTROL_AND_ELO,
            {"pgns", "time_controls", "white_elos", "black_elos"},
        ),
    ],
)
def test_carlini_config_exposes_mode_specific_contract(mode, required):
    config = CarliniPipelineConfig(input_mode=mode)

    assert config.num_lookahead_tokens == 9
    assert config.input_config.required_inputs == frozenset(required)


def test_unimaia_time_info_prompt_style_is_configurable():
    config = UniMaiaPipelineConfig(time_info_prompt_style="legacy")

    assert config.time_info_prompt_style == TimeInfoPromptStyle.LEGACY


def test_published_unimaia_aux_defaults_to_legacy_time_prompt(monkeypatch):
    from chessnets.pipelines import auto_pipeline

    captured = {}

    def fake_create(model_id, device, **kwargs):
        captured.update(model_id=model_id, device=device, **kwargs)
        return object()

    monkeypatch.setattr(auto_pipeline, "create_unimaia_pipeline", fake_create)
    auto_pipeline.PIPELINE_INFOS["unimaia-aux"].factory_fn("cpu")

    assert (
        captured["pipeline_config"].time_info_prompt_style == TimeInfoPromptStyle.LEGACY
    )
