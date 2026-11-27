import json

import pytest

from chessnets.pipelines.auto_pipeline import AutoPipeline


def _write_config(path, architectures):
    (path / "config.json").write_text(
        json.dumps({"architectures": architectures}), encoding="utf-8"
    )


def test_config_architecture_dispatch_passes_source_and_loader_kwargs(tmp_path):
    _write_config(tmp_path, ["ConfigTestModel"])
    calls = {}

    AutoPipeline.register_architecture(
        "ConfigTestModel",
        lambda model_id, device, **kwargs: (
            calls.update(model_id=model_id, device=device, kwargs=kwargs) or "pipeline"
        ),
    )

    assert (
        AutoPipeline.from_pretrained(
            tmp_path,
            device="cpu",
            revision="revision-1",
            token="token-1",
            local_files_only=True,
        )
        == "pipeline"
    )
    assert calls == {
        "model_id": tmp_path,
        "device": "cpu",
        "kwargs": {
            "revision": "revision-1",
            "token": "token-1",
            "local_files_only": True,
        },
    }


@pytest.mark.parametrize(
    "architecture",
    ["Lc0ChessFormerModel", "Lc0SeResNetModel", "BTLeelaZeroNet", "SeLeelaZeroNet"],
)
def test_lc0_architectures_include_pre_0_2_class_names(architecture):
    # Checkpoints converted before 0.2.0 name the old classes in config.json.
    info = AutoPipeline.get_architecture_info(architecture)
    assert info.architecture in {"Lc0ChessFormerModel", "Lc0SeResNetModel"}


def test_exact_id_registration_precedes_config_metadata(tmp_path):
    # Exact registrations must not inspect metadata at all; this is malformed
    # on purpose to catch an accidental config read before the registry lookup.
    tmp_path.joinpath("config.json").write_text("not-json", encoding="utf-8")
    calls = []

    AutoPipeline.register(
        str(tmp_path),
        lambda device, **kwargs: calls.append((device, kwargs)) or "exact",
    )

    assert AutoPipeline.from_pretrained(str(tmp_path), device="cpu") == "exact"
    assert calls == [("cpu", {})]


@pytest.mark.parametrize(
    "contents, message",
    [
        ("[]", "top-level JSON object"),
        ("{}", "non-empty 'architectures' list"),
        ('{"architectures": "Model"}', "non-empty 'architectures' list"),
        ('{"architectures": ["Model", 1]}', "invalid .*architectures.*entries"),
    ],
)
def test_config_architecture_metadata_is_validated(tmp_path, contents, message):
    tmp_path.joinpath("config.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        AutoPipeline.from_pretrained(tmp_path)


def test_config_architecture_dispatch_uses_first_registered_entry(tmp_path):
    _write_config(tmp_path, ["NotRegistered", "ConfigOrderedModel", "AlsoRegistered"])
    calls = []
    AutoPipeline.register_architecture(
        "ConfigOrderedModel",
        lambda model_id, device, **kwargs: calls.append("first") or "first-pipeline",
    )
    AutoPipeline.register_architecture(
        "AlsoRegistered",
        lambda model_id, device, **kwargs: calls.append("second") or "second-pipeline",
    )

    assert AutoPipeline.from_pretrained(tmp_path, device="cpu") == "first-pipeline"
    assert calls == ["first"]


def test_unknown_config_architecture_reports_metadata(tmp_path):
    _write_config(tmp_path, ["MissingModel"])

    with pytest.raises(ValueError, match="MissingModel"):
        AutoPipeline.from_pretrained(tmp_path)


CHAT = "Waterhorse/chessgpt-chat-v1"
BASE = "Waterhorse/chessgpt-base-v1"


@pytest.mark.parametrize(
    "profile_id, checkpoint_id, prompt_format",
    [
        (CHAT, CHAT, None),
        (f"{CHAT}_simple", CHAT, "chat"),
        (f"{CHAT}_simple_original", CHAT, "chat_original"),
        (f"{CHAT}_general_policy", CHAT, "general_policy"),
        (BASE, BASE, "pgn_header"),
        (
            f"{BASE}_with-time-control-and-elo",
            BASE,
            "pgn_header_with_time_control_and_elo",
        ),
    ],
)
def test_chessgpt_profiles_use_curated_pipeline_and_checkpoint(
    monkeypatch, profile_id, checkpoint_id, prompt_format
):
    from chessnets.pipelines.text_generation import pipeline_text_generation as text

    calls = []

    class StubPipeline:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return "pipeline"

    class_name = (
        "TransformersProbabilityPipeline"
        if prompt_format is None
        else "TextGenerationPipeline"
    )
    monkeypatch.setattr(text, class_name, StubPipeline)

    result = AutoPipeline.from_pretrained(
        profile_id,
        device="cpu",
        revision="pinned",
        local_files_only=True,
    )

    expected = {"device": "cpu", "revision": "pinned", "local_files_only": True}
    if prompt_format is not None:
        expected["pipeline_config"] = text.TextGenerationPipelineConfig(
            prompt_format=prompt_format
        )
    assert result == "pipeline"
    assert calls == [(checkpoint_id, expected)]


@pytest.mark.parametrize(
    "profile_id, checkpoint_id, prompt_format",
    [(f"{CHAT}_simple", CHAT, "chat"), (BASE, BASE, "pgn_header")],
)
def test_chessgpt_profiles_pass_choice_constraint_to_pipeline(
    monkeypatch, profile_id, checkpoint_id, prompt_format
):
    from chessnets.pipelines.text_generation import pipeline_text_generation as text

    calls = []

    class StubPipeline:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return "pipeline"

    monkeypatch.setattr(text, "TextGenerationPipeline", StubPipeline)

    result = AutoPipeline.from_pretrained(profile_id, device="cpu", constraint="choice")

    assert result == "pipeline"
    assert calls == [
        (
            checkpoint_id,
            {
                "device": "cpu",
                "constraint": "choice",
                "pipeline_config": text.TextGenerationPipelineConfig(
                    prompt_format=prompt_format
                ),
            },
        )
    ]
