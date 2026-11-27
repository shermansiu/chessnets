"""Pin the exact text each text-generation profile sends to the model.

``data/text_generation_profiles.json`` was recorded from ChessNets 0.1.1 as it
ran with Outlines before 1.2.10, which sent prompts unchanged. Outlines 1.2.10
and later instead wrap a prompt in the tokenizer's chat template, when it has
one. The fakes below can behave either way, and every profile must produce the
recorded text under both, so that prompts never depend on the Outlines version.

Published benchmark results depend on these prompts, so any difference is a
behavior change, not a refactor. Regenerate the file only for a deliberate
prompt change, and record it in the changelog.
"""

from __future__ import annotations

import json
import pathlib
import types

import jinja2
import pytest
import torch

from chessnets.pipelines.auto_pipeline import AutoPipeline
from chessnets.pipelines.carlini import pipeline_carlini as carlini
from chessnets.pipelines.text_generation import pipeline_text_generation as text

EXPECTED = json.loads(
    (
        pathlib.Path(__file__).parent / "data" / "text_generation_profiles.json"
    ).read_text(encoding="utf-8")
)
PGNS = ["1. e4 e5 2. Nf3", '[Event "x"]\n[WhiteElo "1500"]\n\n1. d4']
INPUTS = {
    "pgn+prompt": dict(pgn=PGNS, prompt=["Play well.", "Think."]),
    "tc+elo": dict(
        pgn=PGNS,
        time_control=["300+0", "600+5"],
        white_elo=[1500, 2000],
        black_elo=[1400, 2100],
    ),
}
CHAT, BASE = "Waterhorse/chessgpt-chat-v1", "Waterhorse/chessgpt-base-v1"
PROFILES = {
    "chat-v1": (CHAT, "pgn+prompt"),
    "chat-v1_simple": (f"{CHAT}_simple", "pgn+prompt"),
    "chat-v1_simple_original": (f"{CHAT}_simple_original", "pgn+prompt"),
    "chat-v1_general_policy": (f"{CHAT}_general_policy", "tc+elo"),
    "base-v1": (BASE, "pgn+prompt"),
    "base-v1_tc_elo": (f"{BASE}_with-time-control-and-elo", "tc+elo"),
    # How UniMaia loads its archived ChessGPT fine-tunes.
    "unimaia-local": (None, "pgn+prompt"),
}
CASES = [(name, False) for name in PROFILES] + [
    (name, True) for name in PROFILES if name != "chat-v1"
]
OUTLINES_BEHAVIORS = ["before-1.2.10", "from-1.2.10"]


def _clean(kwargs):
    return {
        key: str(value) if isinstance(value, torch.dtype) else value
        for key, value in kwargs.items()
    }


def install_fakes(monkeypatch, outlines_behavior):
    """Replace Transformers and Outlines with fakes that record their calls.

    The recorded prompts are the text the model would see: from 1.2.10,
    Outlines renders the tokenizer's chat template around a string prompt, as
    one user message with a generation prompt, whenever there is a template.
    """
    recorded = []

    class Tokenizer:
        pad_token_id = None
        eos_token_id = 0
        # Stands for a template the checkpoint ships with.
        chat_template = "Human: {{ messages[0]['content'] }}\nAssistant:"

    class Model:
        def __init__(self):
            self.config = types.SimpleNamespace(
                pad_token_id=None, max_position_embeddings=7
            )
            self.generation_config = types.SimpleNamespace(pad_token_id=None)

        def to(self, device):
            recorded.append(["to", str(device)])
            return self

    def load(kind, result):
        def from_pretrained(model_id, **kwargs):
            recorded.append([kind, model_id, _clean(kwargs)])
            return result()

        return types.SimpleNamespace(from_pretrained=from_pretrained)

    def seen_by_model(tokenizer, prompt):
        if outlines_behavior == "from-1.2.10" and tokenizer.chat_template:
            return jinja2.Template(tokenizer.chat_template).render(
                messages=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
            )
        return prompt

    class Regex:
        def __init__(self, pattern):
            self.desc = ["regex", pattern]

    class Choice:
        def __init__(self, items):
            self.desc = ["choice", list(items)]

    class Generator:
        def __init__(self, model, output_type):
            self.tokenizer = model.tokenizer
            self.desc = getattr(output_type, "desc", None) or output_type.__name__
            recorded.append(["generator", self.desc])

        def batch(self, prompts, **kwargs):
            seen = [seen_by_model(self.tokenizer, prompt) for prompt in prompts]
            recorded.append(["batch", seen, _clean(kwargs)])
            if self.desc == "ChessResponse":
                return ['{"moves": [{"move": "e4", "probability": 1.0}]}'] * len(
                    prompts
                )
            return ["1. e4"] * len(prompts)

        def __call__(self, prompt, **kwargs):
            seen = seen_by_model(self.tokenizer, prompt)
            recorded.append(["call", seen, _clean(kwargs)])
            return self.desc[1][0]

    monkeypatch.setattr(
        text,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=load("tok", Tokenizer),
            AutoModelForCausalLM=load("model", Model),
        ),
    )
    monkeypatch.setattr(
        text,
        "outlines",
        types.SimpleNamespace(
            from_transformers=lambda model, tokenizer: types.SimpleNamespace(
                model=model, tokenizer=tokenizer
            ),
            Generator=Generator,
            types=types.SimpleNamespace(Regex=Regex, Choice=Choice),
            models=types.SimpleNamespace(
                OpenAI=type("OpenAI", (), {}), AsyncOpenAI=type("AsyncOpenAI", (), {})
            ),
        ),
    )
    return recorded


def observe(recorded, load_pipeline, input_kind):
    """Load a pipeline, predict, and return everything the fakes saw."""
    pipeline = load_pipeline()
    observed = {
        "init_calls": list(recorded),
        "temperature": pipeline.temperature,
        "required": sorted(pipeline.input_config.required_inputs),
        "pad": [
            pipeline.model.tokenizer.pad_token_id,
            pipeline.model.model.config.pad_token_id,
        ],
    }
    recorded.clear()
    pipeline.make_prediction(**INPUTS[input_kind], batch_size=8)
    observed["predict_calls"] = list(recorded)
    return observed


def _load(profile_id, **kwargs):
    if profile_id is not None:
        return AutoPipeline.from_pretrained(profile_id, device="cpu", **kwargs)
    return text.TextGenerationPipeline.from_pretrained(
        "local/dir",
        device="cpu",
        dtype=None,
        attn_implementation=None,
        prompt_template=text.END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE,
        pipeline_config=text.TextGenerationPipelineConfig(prompt_format="pgn_header"),
        **kwargs,
    )


@pytest.mark.parametrize("outlines_behavior", OUTLINES_BEHAVIORS)
@pytest.mark.parametrize(
    "name, choice", CASES, ids=[f"{n}{'+choice' if c else ''}" for n, c in CASES]
)
def test_profile_sends_the_recorded_text(monkeypatch, name, choice, outlines_behavior):
    recorded = install_fakes(monkeypatch, outlines_behavior)
    profile_id, input_kind = PROFILES[name]
    kwargs = {"constraint": "choice"} if choice else {}

    observed = observe(recorded, lambda: _load(profile_id, **kwargs), input_kind)

    assert observed == EXPECTED[name + ("+choice" if choice else "")]


@pytest.mark.parametrize("mode", ["default", "prompt", "time_control_and_elo"])
def test_carlini_mode_sends_the_recorded_prompts(tmp_path, mode):
    class Client:
        def __init__(self, **kwargs):
            pass

    pipeline = carlini.CarliniPipeline(
        model_id="x",
        client_factory=Client,
        cache_path_template=str(tmp_path / "{model_id}.json"),
        pipeline_config=carlini.CarliniPipelineConfig(input_mode=mode),
    )
    prompts = []
    pipeline.forward = lambda prompt, num_tokens: prompts.append(prompt) or " e4"
    inputs = {
        "default": {},
        "prompt": {"prompts": ["P1", "P2"]},
        "time_control_and_elo": {
            "time_controls": ["300+0", "600+5"],
            "white_elos": [1500, 2000],
            "black_elos": [1400, 2100],
        },
    }[mode]

    pipeline.make_prediction(PGNS, **inputs)

    expected = EXPECTED[f"carlini-{mode}"]
    assert prompts == expected["prompts"]
    assert sorted(pipeline.input_config.required_inputs) == expected["required"]


def test_positional_inputs_follow_the_format_order(monkeypatch):
    # ChessEval passes input columns positionally after the PGNs.
    recorded = install_fakes(monkeypatch, "from-1.2.10")
    pipeline = _load(f"{CHAT}_general_policy")
    by_keyword = pipeline.make_prediction(**INPUTS["tc+elo"])
    keyword_prompts = recorded[-1][1]
    tc = INPUTS["tc+elo"]
    by_position = pipeline.make_prediction(
        PGNS, tc["time_control"], tc["white_elo"], tc["black_elo"]
    )

    assert recorded[-1][1] == keyword_prompts
    assert len(by_position.predictions) == len(by_keyword.predictions)
    with pytest.raises(TypeError, match="positional"):
        pipeline.make_prediction(PGNS, *([None] * 4))
