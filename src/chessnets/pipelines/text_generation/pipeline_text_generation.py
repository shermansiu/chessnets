"""Text-generation chess pipelines backed by Transformers, Outlines, or OpenAI.

``TextGenerationPipeline`` asks a causal language model such as ChessGPT to
write the next move, in one of the prompt formats of :class:`TextPromptFormat`.
The ``*ProbabilityPipeline`` classes ask a model for a JSON distribution over
the legal moves instead.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import logging
import random
import textwrap
import types
import typing as tp

import chess
import dacite
import torch
import torch.utils.data

from chessnets._optional import optional_module
from chessnets.utils import board_history

from ..base import (
    Pipeline,
    PipelineInputConfig,
    bind_positional_inputs,
    extract_domain_inputs,
)
from ..outputs import (
    ChessMove,
    ChessMovePredictionDataset,
    ChessResponse,
    LLMChessMovePrediction,
)
from ..pipeline_utils import (
    DEFAULT_PARALLEL_THRESHOLD,
    adaptive_process_map,
    asyncio_run,
    iter_dataset_batches,
    progress,
    resolve_num_workers,
)

jinja2_sandbox = optional_module("jinja2.sandbox", "text-generation")
openai = optional_module("openai", "text-generation")
outlines = optional_module("outlines", "text-generation")
transformers = optional_module("transformers", "text-generation")
IPython = types.SimpleNamespace(display=optional_module("IPython.display", "notebook"))


DEFAULT_OUTLINES_PROMPT_TEMPLATE = textwrap.dedent(
    """{prompt}

    PGN: {pgn}

    FEN: {fen}

    Legal moves: {legal_moves_str}

    Determine the probability of playing each legal chess move in the given position. Respond in JSON format as a list of objects with `move` and `probability` fields, where `move` is in the SAN format. Ensure the probabilities sum to 1 and only include legal moves.
    """.strip()
)


class _PromptDataset(torch.utils.data.Dataset):
    """Prompts filled in from a template with the PGN, FEN and legal moves."""

    def __init__(
        self,
        pgns: list[str] | str,
        prompts: list[str] | str,
        should_shuffle_legal_moves: bool = False,
        prompt_template: str = DEFAULT_OUTLINES_PROMPT_TEMPLATE,
        *,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
        if isinstance(prompts, str):
            prompts = [prompts]
        assert len(pgns) == len(prompts), "PGNs and prompts must have the same length."
        self.histories: list[board_history.BoardHistory] = adaptive_process_map(
            board_history.BoardHistory.from_pgn,
            pgns,
            parallel_threshold=parallel_threshold,
            max_workers=num_workers,
            chunksize=16,
            desc="Converting PGNs to board histories",
            leave=False,
        )
        self.legal_moves: list[chess.Move] = adaptive_process_map(
            self.history_to_legal_moves,
            self.histories,
            [should_shuffle_legal_moves] * len(self.histories),
            parallel_threshold=parallel_threshold,
            max_workers=num_workers,
            chunksize=16,
            desc="Obtaining legal moves for board histories",
            leave=False,
        )
        self.consolidated_prompts: list[str] = adaptive_process_map(
            self.create_consolidated_prompt,
            prompts,
            pgns,
            self.histories,
            self.legal_moves,
            [prompt_template] * len(prompts),
            parallel_threshold=parallel_threshold,
            max_workers=num_workers,
            chunksize=16,
            desc="Creating consolidated prompts",
            leave=False,
        )
        self.should_shuffle_legal_moves = should_shuffle_legal_moves

    def __len__(self):
        return len(self.consolidated_prompts)

    @staticmethod
    def history_to_legal_moves(
        history: board_history.BoardHistory, should_shuffle: bool
    ) -> list[chess.Move]:
        legal_moves = list(history.active_board.legal_moves)
        if should_shuffle:
            random.shuffle(legal_moves)
        return legal_moves

    @staticmethod
    def create_consolidated_prompt(
        prompt: str,
        pgn: str,
        history: board_history.BoardHistory,
        legal_moves: list[chess.Move],
        prompt_template: str,
    ) -> str:
        legal_moves_str = " ".join([history.active_board.san(m) for m in legal_moves])
        fen = history.active_board.fen()
        return prompt_template.format(
            prompt=prompt, pgn=pgn, fen=fen, legal_moves_str=legal_moves_str
        )

    def __getitem__(self, idx: int):
        consolidated_prompt = self.consolidated_prompts[idx]
        history = self.histories[idx]
        legal_moves = self.legal_moves[idx]
        return consolidated_prompt, history, legal_moves

    @staticmethod
    def collate_fn(batch):
        consolidated_prompts, histories, legal_moves_list = zip(*batch)
        return list(consolidated_prompts), list(histories), list(legal_moves_list)


# We keep the time control in the event name "Classical" because the LLM keeps making throwaway moves for faster time controls because of the time pressure
WATERHORSE_CHESSGPT_GENERAL_POLICY_PROMPT_TEMPLATE = """In the following chess game, you play {colour}:
[Event "Rated Classical game https://lichess.org/tournament/xxxx"]
[Date "2017.04.01"]
[Round "-"]
[White "???"]
[Black "???"]
[Result "{win_result}"]
[WhiteElo "{white_elo}"]
[BlackElo "{black_elo}"]
[WhiteRatingDiff "??"]
[BlackRatingDiff "??"]
[ECO "??"]
[Opening "??"]
[TimeControl "{time_control}"]
[Termination "Time forfeit"]

{pgn}

Make the next move."""


class _GeneralPolicyDataset(torch.utils.data.Dataset):
    """ChessGPT general-policy prompts: a PGN header with ratings and time control."""

    def __init__(
        self,
        pgn: list[str] | str,
        time_control: list[str] | str,
        white_elo: list[int] | int,
        black_elo: list[int] | int,
        prompt_template: str = WATERHORSE_CHESSGPT_GENERAL_POLICY_PROMPT_TEMPLATE,
        *,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    ):
        if isinstance(pgn, str):
            pgn = [pgn]
        if isinstance(time_control, str):
            time_control = [time_control]
        if isinstance(white_elo, int):
            white_elo = [white_elo]
        if isinstance(black_elo, int):
            black_elo = [black_elo]
        assert len(pgn) == len(time_control) == len(white_elo) == len(black_elo), (
            "PGNs, time controls, white ELOs, and black ELOs must have the same length."
        )
        self.pgns = pgn
        self.time_controls = time_control
        self.white_elos = white_elo
        self.black_elos = black_elo
        self.prompt_template = prompt_template
        self.histories: list[board_history.BoardHistory] = adaptive_process_map(
            board_history.BoardHistory.from_pgn,
            pgn,
            parallel_threshold=parallel_threshold,
            max_workers=num_workers,
            chunksize=16,
            desc="Converting PGNs to board histories",
            leave=False,
        )

    def __len__(self):
        return len(self.pgns)

    @staticmethod
    def create_prompt(
        pgn: str,
        history: board_history.BoardHistory,
        time_control: str,
        white_elo: int,
        black_elo: int,
        prompt_template: str,
    ) -> str:
        colour = "white" if history.active_board.turn == chess.WHITE else "black"
        win_result = "1-0" if colour == "white" else "0-1"
        pgn = pgn.split("\n\n", 1)[-1]
        return prompt_template.format(
            colour=colour,
            win_result=win_result,
            time_control=time_control,
            white_elo=white_elo,
            black_elo=black_elo,
            pgn=pgn,
        )

    def __getitem__(self, idx: int):
        history = self.histories[idx]
        pgn = self.pgns[idx]
        time_control = self.time_controls[idx]
        white_elo = self.white_elos[idx]
        black_elo = self.black_elos[idx]

        prompt = self.create_prompt(
            pgn=pgn,
            history=history,
            time_control=time_control,
            white_elo=white_elo,
            black_elo=black_elo,
            prompt_template=self.prompt_template,
        )
        legal_moves = list(history.active_board.legal_moves)
        return prompt, history, legal_moves

    @staticmethod
    def collate_fn(batch):
        prompts, histories, legal_moves_list = zip(*batch)
        return list(prompts), list(histories), list(legal_moves_list)


class OutlinesProbabilityPipeline(Pipeline):
    """Ask any Outlines model for a JSON probability for each legal move."""

    def __init__(
        self,
        model: outlines.Model,
        temperature: float | None = 0.0,
        prompt_template: str = DEFAULT_OUTLINES_PROMPT_TEMPLATE,
        input_config: PipelineInputConfig | None = None,
    ):
        super().__init__(input_config=input_config)
        self.model = model
        self.temperature = temperature
        self.generator = self.get_generator_for_model(model)
        self.prompt_template = prompt_template

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(required=("pgns", "prompts"))

    @staticmethod
    def get_generator_for_model(model: outlines.Model, *args, **kwargs):
        return outlines.Generator(model, ChessResponse)

    def forward(self, consolidated_prompts: list[str], *args, **kwargs):
        if isinstance(self.model, outlines.models.OpenAI):
            results = [
                self.generator(prompt, temperature=self.temperature, **kwargs)
                for prompt in consolidated_prompts
            ]
        elif isinstance(self.model, outlines.models.AsyncOpenAI):

            async def async_forward(prompt) -> ChessResponse:
                return await self.generator(
                    prompt, temperature=self.temperature, **kwargs
                )

            async def get_results(prompts) -> list[ChessResponse]:
                return await asyncio.gather(
                    *[async_forward(prompt) for prompt in prompts]
                )

            results = get_results(consolidated_prompts)
        else:
            results = self.generator.batch(
                consolidated_prompts, temperature=self.temperature, **kwargs
            )
        if isinstance(results, list):
            return results
        return asyncio_run(results)

    def forward_batch(
        self,
        consolidated_prompts: list[str],
        histories: list[board_history.BoardHistory],
        legal_moves: list[list[chess.Move]],
    ):
        """Generate one response per prompt, with position context available."""
        return self.forward(consolidated_prompts)

    def make_prediction(
        self,
        pgn: str | list[str],
        prompt: str | list[str],
        *args,
        should_shuffle_legal_moves: bool = False,
        return_histories: bool = False,
        tqdm_leave: bool = False,
        batch_size: int = 16,
        persistent_workers: bool = False,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not hasattr(self, "input_config"):
            self.input_config = self.default_input_config()
        self.input_config.validate(
            pgns=pgn, prompts=prompt, **extract_domain_inputs(kwargs)
        )
        item_count = 1 if isinstance(pgn, str) else len(pgn)
        resolved_workers = resolve_num_workers(
            item_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )
        dataset = _PromptDataset(
            pgns=pgn,
            prompts=prompt,
            should_shuffle_legal_moves=should_shuffle_legal_moves,
            prompt_template=self.prompt_template,
            num_workers=resolved_workers,
            parallel_threshold=parallel_threshold,
        )
        return self._make_predictions_from_dataset(
            dataset,
            return_histories=return_histories,
            tqdm_leave=tqdm_leave,
            batch_size=batch_size,
            persistent_workers=persistent_workers,
            num_workers=resolved_workers,
            parallel_threshold=parallel_threshold,
        )

    def _make_predictions_from_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        return_histories: bool,
        tqdm_leave: bool,
        batch_size: int,
        persistent_workers: bool,
        num_workers: int | None,
        parallel_threshold: int,
    ) -> ChessMovePredictionDataset:
        """Run the shared batched inference and post-processing path."""
        raw_outputs = []
        for consolidated_prompts, histories, all_legal_moves in progress(
            iter_dataset_batches(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
            ),
            desc="Making predictions",
            total=(len(dataset) + batch_size - 1) // batch_size,
            leave=tqdm_leave,
        ):
            try:
                responses = self.forward_batch(
                    consolidated_prompts,
                    histories,
                    all_legal_moves,
                )
                if len(responses) != len(histories):
                    raise ValueError(
                        "Text-generation backend returned a different number "
                        "of responses than inputs"
                    )
                raw_outputs.extend(
                    zip(
                        responses,
                        histories,
                        all_legal_moves,
                        [return_histories] * len(histories),
                    )
                )
            except Exception as e:
                if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
                    raise e
                logging.error("Error processing model response: %s", e)
                raw_outputs.extend(
                    (
                        None,
                        history,
                        legal_moves,
                        return_histories,
                    )
                    for history, legal_moves in zip(histories, all_legal_moves)
                )
        if raw_outputs:
            predictions = adaptive_process_map(
                self.post_process_predictions,
                *zip(*raw_outputs),
                parallel_threshold=parallel_threshold,
                max_workers=num_workers,
                desc="Collecting predictions",
                leave=tqdm_leave,
                chunksize=64,
            )
        else:
            predictions = []
        return ChessMovePredictionDataset[LLMChessMovePrediction](
            predictions=predictions
        )

    @classmethod
    def post_process_success(
        cls,
        response: str,
        history: board_history.BoardHistory,
        legal_moves: list[chess.Move],
        return_histories: bool = False,
    ):
        try:
            chess_response = dacite.from_dict(
                data_class=ChessResponse, data=json.loads(response)
            )
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON object")
            return cls.post_process_failure(
                history, legal_moves, return_histories=return_histories
            )
        best_move_san, best_move_uci, preds_list = (
            chess_response.to_best_moves_and_preds_list(
                history.active_board, rescale_excess=True, fill_missing=True
            )
        )
        if return_histories:
            best_move = chess.Move.from_uci(best_move_uci)
            history.append(best_move)
        return LLMChessMovePrediction(
            board_history=history if return_histories else None,
            best_move_san=best_move_san,
            best_move_uci=best_move_uci,
            preds_list=preds_list,
            raw_response=response,
        )

    @classmethod
    def post_process_failure(
        cls,
        history: board_history.BoardHistory,
        legal_moves: list[chess.Move],
        return_histories: bool = False,
    ):
        # If there's an error, just pick a random legal move
        best_move = random.choice(legal_moves)
        best_move_san = history.active_board.san(best_move)
        best_move_uci = str(best_move)
        preds_list = {m: 1 / len(legal_moves) for m in legal_moves}
        return LLMChessMovePrediction(
            board_history=history if return_histories else None,
            best_move_san=best_move_san,
            best_move_uci=best_move_uci,
            preds_list=preds_list,
            raw_response=None,
        )

    @classmethod
    def post_process_predictions(
        cls,
        response: str | None,
        history: board_history.BoardHistory,
        legal_moves: list[chess.Move],
        return_histories: bool = False,
    ):
        if response is not None:
            return cls.post_process_success(
                response, history, legal_moves, return_histories=return_histories
            )
        assert legal_moves is not None
        return cls.post_process_failure(
            history, legal_moves, return_histories=return_histories
        )


MAX_RETRIES = 5


class OpenAIProbabilityPipeline(OutlinesProbabilityPipeline):
    """:class:`OutlinesProbabilityPipeline` for a hosted OpenAI-compatible model."""

    def __init__(
        self,
        model_id: str,
        temperature: float = 1.0,
        input_config: PipelineInputConfig | None = None,
    ):
        super().__init__(
            outlines.from_openai(
                openai.OpenAI(max_retries=MAX_RETRIES), model_name=model_id
            ),
            temperature=temperature,
            input_config=input_config,
        )

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(required=("pgns", "prompts"))

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: tp.Any
    ) -> tp.Self:
        """Select a hosted OpenAI model by its model ID."""
        return cls(model_id=pretrained_model_name_or_path, **kwargs)


class AsyncOpenAIProbabilityPipeline(OutlinesProbabilityPipeline):
    """:class:`OpenAIProbabilityPipeline` with concurrent requests."""

    def __init__(
        self,
        model_id: str,
        temperature: float = 1.0,
        input_config: PipelineInputConfig | None = None,
    ):
        super().__init__(
            outlines.from_openai(
                openai.AsyncOpenAI(max_retries=MAX_RETRIES), model_name=model_id
            ),
            temperature=temperature,
            input_config=input_config,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: tp.Any
    ) -> tp.Self:
        """Select a hosted OpenAI model by its model ID."""
        return cls(model_id=pretrained_model_name_or_path, **kwargs)


def _default_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401 -- probe optional dependency availability
    except ImportError:
        return "sdpa"
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return "flash_attention_2"
    return "sdpa"


PAPER_CHESSGPT_TEMPLATE = "{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim + '<|endoftext|>' if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}\n{%- for message in messages -%}\n{%- if message['role'] == 'user' -%}\nHuman 0: {{ message['content']|trim -}}\n{%- if not loop.last -%}\n<|endoftext|>\n{%- endif -%}\n{%- elif message['role'] == 'assistant' -%}\nHuman 1: {{ message['content']|trim -}}\n{%- if not loop.last -%}\n<|endoftext|>\n{%- endif -%}\n{%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}\n<|endoftext|>Human 1: {% endif -%}\n"
NEWLINE_CHESSGPT_TEMPLATE = "{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim + '<|endoftext|>' if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}\n{%- for message in messages -%}\n{%- if message['role'] == 'user' -%}\nHuman 0: {{ '\n'~message['content']|trim -}}\n{%- if not loop.last -%}\n<|endoftext|>\n{%- endif -%}\n{%- elif message['role'] == 'assistant' -%}\nHuman 1: {{ '\n'~message['content']|trim -}}\n{%- if not loop.last -%}\n<|endoftext|>\n{%- endif -%}\n{%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}\n<|endoftext|>Human 1: {% endif -%}\n"
CHESSGPT_SYSTEM_PROMPT = "A friendly, helpful chat between some humans."


def render_chat_prompt_template(
    chat_template: str,
    prompt_template: str,
    system_prompt: str = CHESSGPT_SYSTEM_PROMPT,
) -> str:
    """Render a chat template around a prompt template, keeping its fields.

    The result is one system message and one user message, the user message
    being ``prompt_template`` with its ``{prompt}``-style fields still to be
    filled in, followed by the generation prompt.
    """
    environment = jinja2_sandbox.SandboxedEnvironment()
    return environment.from_string(chat_template).render(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_template},
        ],
        add_generation_prompt=True,
    )


class TransformersProbabilityPipeline(OutlinesProbabilityPipeline):
    """:class:`OutlinesProbabilityPipeline` for a local Transformers causal LM.

    ChessNets builds the whole prompt itself, so the model sees the same text
    with every Outlines version. With a ``chat_template``, the prompt template
    is rendered as a system message (:data:`CHESSGPT_SYSTEM_PROMPT`) and a
    user message, as :func:`render_chat_prompt_template` does. The tokenizer
    is given no chat template, so Outlines never applies one again; Outlines
    1.2.10 and later otherwise wrap every prompt in it.
    """

    auto_batch_sizes = {"cpu": 4, "cuda": 8}

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.0,
        prompt_template: str = DEFAULT_OUTLINES_PROMPT_TEMPLATE,
        chat_template: str | None = None,
        input_config: PipelineInputConfig | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        model_kwargs: dict[str, tp.Any] | None = None,
        tokenizer_kwargs: dict[str, tp.Any] | None = None,
        attn_implementation: str | None = "auto",
    ):
        model, tokenizer = self.load_model_and_tokenizer(
            model_id=model_id,
            device=device,
            dtype=dtype,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            attn_implementation=attn_implementation,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.eos_token_id
        else:
            if (
                hasattr(model.config, "pad_token_id")
                and model.config.pad_token_id is None
            ):
                model.config.pad_token_id = tokenizer.pad_token_id
            if (
                hasattr(model.generation_config, "pad_token_id")
                and model.generation_config.pad_token_id is None
            ):
                model.generation_config.pad_token_id = tokenizer.pad_token_id
        if chat_template is not None:
            prompt_template = render_chat_prompt_template(
                chat_template, prompt_template
            )
        # Outlines 1.2.10 and later apply the tokenizer's chat template to
        # string prompts, and earlier versions do not. Without one, every
        # version sends the prompt rendered above unchanged.
        tokenizer.chat_template = None
        if temperature == 0.0:
            temperature = None
        super().__init__(
            outlines.from_transformers(
                model,
                tokenizer,
            ),
            temperature=temperature,
            prompt_template=prompt_template,
            input_config=input_config,
        )

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(required=("pgns", "prompts"))

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: tp.Any
    ) -> tp.Self:
        """Load a Transformers checkpoint and construct the text pipeline."""
        return cls(model_id=pretrained_model_name_or_path, **kwargs)

    def load_model_and_tokenizer(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        model_kwargs: dict[str, tp.Any] | None = None,
        tokenizer_kwargs: dict[str, tp.Any] | None = None,
        attn_implementation: str | None = "auto",
    ):
        """Load the tokenizer and model.

        ``attn_implementation="auto"`` picks FlashAttention 2 when it is
        installed and the GPU supports it, and SDPA otherwise; ``None`` leaves
        the choice to Transformers.
        """
        common_kwargs = {
            "revision": revision,
            "cache_dir": cache_dir,
            "token": token,
            "local_files_only": local_files_only,
        }
        resolved_tokenizer_kwargs = {**common_kwargs, **(tokenizer_kwargs or {})}
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, **resolved_tokenizer_kwargs
        )
        if attn_implementation == "auto":
            attn_implementation = _default_attn_implementation()
        resolved_model_kwargs = {**common_kwargs, **(model_kwargs or {})}
        resolved_model_kwargs.setdefault("dtype", dtype)
        if attn_implementation is not None:
            resolved_model_kwargs.setdefault("attn_implementation", attn_implementation)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, **resolved_model_kwargs
        ).to(device=device)
        return model, tokenizer

    def forward(self, consolidated_prompts: list[str], *args, **kwargs):
        if self.temperature is not None:
            kwargs["do_sample"] = True
        kwargs["max_new_tokens"] = self.model.model.config.max_position_embeddings
        return super().forward(consolidated_prompts, *args, **kwargs)


DEFAULT_BASIC_PROMPT_TEMPLATE = (
    "{prompt}\n\nPGN: {pgn}\n\nFEN: {fen}\n\nLegal moves: {legal_moves_str}"
)
WATERHORSE_CHESSGPT_PROMPT_TEMPLATE = "A friendly, helpful chat between some humans.<|endoftext|>Human 0: {prompt}\n\nPGN: {pgn}\n\nJust write your move."
WATERHORSE_CHESSGPT_PROMPT_TEMPLATE_NEWLINE = "A friendly, helpful chat between some humans.<|endoftext|>Human 0: \n{prompt}\n\nPGN: {pgn}\n\nJust write your move."
WATERHORSE_CHESSGPT_PROMPT_TEMPLATE_RAW = (
    "{prompt}\n\nPGN: {pgn}\n\nJust write your move."
)
NUMBERED_MOVE_REGEX = (
    r"[1-9][0-9]*\.(?:\.\.)? "
    r"(?:(?:(?:[NBRQK])?(?:[a-h])?(?:[1-8])?x?(?:[a-h][1-8])"
    r"(?:=[NBRQ])?(?:\+|#)?)|(?:O-O(?:-O)))"
)
PGN_CONTINUATION_MOVE_REGEX = (
    r"(?:(?: ?[1-9][0-9]*\.(?:\.\.)?)? )?"
    r"(?:(?:(?:[NBRQK])?(?:[a-h])?(?:[1-8])?x?(?:[a-h][1-8])"
    r"(?:=[NBRQ])?(?:\+|#)?)|(?:O-O(?:-O)))"
)


PGN_HEADER_PROMPT_TEMPLATE = "{prompt}\n\n{pgn}"
END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE = "{prompt}<|endoftext|>{pgn}"
DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO = """[White "???"]\n[Black "???"]\n[WhiteElo "{white_elo}"]\n[BlackElo "{black_elo}"]\n[TimeControl "{time_control}"]\n\n"""


class _TimeControlAndEloDataset(torch.utils.data.Dataset):
    """PGN-continuation prompts whose header gives the ratings and time control."""

    def __init__(
        self,
        pgns: list[str] | str,
        time_control: list[str] | str,
        white_elo: list[int] | int,
        black_elo: list[int] | int,
        pgn_header_override: str | None = None,
        *,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
        if isinstance(time_control, str):
            time_control = [time_control]
        if isinstance(white_elo, int):
            white_elo = [white_elo]
        if isinstance(black_elo, int):
            black_elo = [black_elo]
        assert len(pgns) == len(time_control) == len(white_elo) == len(black_elo), (
            "All input lists must have the same length"
        )

        self.histories: list[board_history.BoardHistory] = adaptive_process_map(
            board_history.BoardHistory.from_pgn,
            pgns,
            parallel_threshold=parallel_threshold,
            max_workers=num_workers,
            chunksize=16,
            desc="Converting PGNs to board histories",
            leave=False,
        )
        if pgn_header_override is not None:
            self.pgn_header = pgn_header_override
        else:
            self.pgn_header = DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO
        self.prompts: list[str] = adaptive_process_map(
            self.create_prompt,
            pgns,
            time_control,
            white_elo,
            black_elo,
            [self.pgn_header] * len(self.histories),
            parallel_threshold=parallel_threshold,
            max_workers=num_workers,
            chunksize=16,
            desc="Creating prompts",
            leave=False,
        )

    def __len__(self):
        return len(self.prompts)

    @staticmethod
    def create_prompt(
        pgn: str,
        time_control: str,
        white_elo: int,
        black_elo: int,
        pgn_header: str = DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO,
    ) -> str:
        prompt = (
            pgn_header.format(
                time_control=time_control,
                white_elo=white_elo,
                black_elo=black_elo,
            )
            + pgn.split("\n\n")[-1]
        )
        return prompt

    def __getitem__(self, idx: int):
        history = self.histories[idx]
        prompt = self.prompts[idx]
        legal_moves = list(history.active_board.legal_moves)
        return prompt, history, legal_moves

    @staticmethod
    def collate_fn(batch):
        prompts, histories, legal_moves_list = zip(*batch)
        return list(prompts), list(histories), list(legal_moves_list)


def numbered_move_choices(
    history: board_history.BoardHistory,
    legal_moves: list[chess.Move],
) -> list[str]:
    """Return each legal move as a numbered move, such as ``"2... Nc6"``."""
    board = history.active_board
    dots = "." if board.turn == chess.WHITE else "..."
    prefix = f"{board.fullmove_number}{dots} "
    return [prefix + board.san(move) for move in legal_moves]


def pgn_continuation_move_choices(
    history: board_history.BoardHistory,
    legal_moves: list[chess.Move],
) -> list[str]:
    """Return every spelling of each legal move that continues a PGN.

    These are the strings :data:`PGN_CONTINUATION_MOVE_REGEX` accepts: the move
    with or without its number, and with or without a leading space.
    """
    board = history.active_board
    dots = "." if board.turn == chess.WHITE else "..."
    prefix = f"{board.fullmove_number}{dots} "
    choices = []
    for move in legal_moves:
        san = board.san(move)
        choices.extend((san, f" {san}", prefix + san, " " + prefix + san))
    return choices


class TextPromptFormat(str, enum.Enum):
    """How :class:`TextGenerationPipeline` asks the model for a move.

    Each format fixes the default prompt template and chat template, the
    inputs the pipeline takes, and the move pattern the output must match.
    """

    CHAT = "chat"
    """A ChessGPT chat turn: the prompt, the PGN, and "Just write your move."."""

    CHAT_ORIGINAL = "chat_original"
    """Like ``CHAT``, but the template holds only the user's message."""

    GENERAL_POLICY = "general_policy"
    """ChessGPT's general-policy prompt: a PGN header with both ratings and the
    time control, for the side to move."""

    PGN_HEADER = "pgn_header"
    """Plain completion of the prompt followed by the PGN, for base models."""

    PGN_HEADER_WITH_TIME_CONTROL_AND_ELO = "pgn_header_with_time_control_and_elo"
    """Plain completion of a PGN whose header gives the ratings and time control."""


_PROMPT_INPUTS = ("prompts",)
_TIME_CONTROL_AND_ELO_INPUTS = ("time_controls", "white_elos", "black_elos")
_CHAT_FORMATS = frozenset(
    {
        TextPromptFormat.CHAT,
        TextPromptFormat.CHAT_ORIGINAL,
        TextPromptFormat.GENERAL_POLICY,
    }
)
_DEFAULT_PROMPT_TEMPLATES = {
    TextPromptFormat.CHAT: WATERHORSE_CHESSGPT_PROMPT_TEMPLATE,
    TextPromptFormat.CHAT_ORIGINAL: WATERHORSE_CHESSGPT_PROMPT_TEMPLATE_RAW,
    TextPromptFormat.GENERAL_POLICY: WATERHORSE_CHESSGPT_GENERAL_POLICY_PROMPT_TEMPLATE,
    TextPromptFormat.PGN_HEADER: PGN_HEADER_PROMPT_TEMPLATE,
    # Unused: the header comes from DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO.
    TextPromptFormat.PGN_HEADER_WITH_TIME_CONTROL_AND_ELO: PGN_HEADER_PROMPT_TEMPLATE,
}


@dataclasses.dataclass(frozen=True)
class TextGenerationPipelineConfig:
    """Options for :class:`TextGenerationPipeline`.

    ``constraint="regex"`` constrains generation to the format's move pattern;
    ``"choice"`` restricts it to the legal moves in each position, one
    generation call per position.
    """

    prompt_format: TextPromptFormat = TextPromptFormat.CHAT
    constraint: tp.Literal["regex", "choice"] = "regex"

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_format, TextPromptFormat):
            try:
                object.__setattr__(
                    self, "prompt_format", TextPromptFormat(self.prompt_format)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported text prompt format: {self.prompt_format!r}"
                ) from exc
        if self.constraint not in {"regex", "choice"}:
            raise ValueError("constraint must be either 'regex' or 'choice'")

    @property
    def input_names(self) -> tuple[str, ...]:
        """The inputs after the PGNs, in the order they may be passed positionally."""
        if self.prompt_format in {
            TextPromptFormat.GENERAL_POLICY,
            TextPromptFormat.PGN_HEADER_WITH_TIME_CONTROL_AND_ELO,
        }:
            return _TIME_CONTROL_AND_ELO_INPUTS
        return _PROMPT_INPUTS

    @property
    def input_config(self) -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(required=("pgns", *self.input_names))

    @property
    def default_prompt_template(self) -> str:
        return _DEFAULT_PROMPT_TEMPLATES[self.prompt_format]

    @property
    def default_chat_template(self) -> str | None:
        """ChessGPT's chat template for chat formats; none for completion."""
        return PAPER_CHESSGPT_TEMPLATE if self.prompt_format in _CHAT_FORMATS else None

    @property
    def move_regex(self) -> str:
        if self.prompt_format in _CHAT_FORMATS:
            return NUMBERED_MOVE_REGEX
        return PGN_CONTINUATION_MOVE_REGEX


_FORMAT_DEFAULT: tp.Any = object()


class TextGenerationPipeline(TransformersProbabilityPipeline):
    """Ask a causal language model, such as ChessGPT, to write the next move.

    Unlike :class:`TransformersProbabilityPipeline`, the model writes a single
    move rather than a JSON distribution, which suits models such as ChessGPT
    and is faster. :class:`TextGenerationPipelineConfig` selects the prompt
    format and how generation is constrained.

    The chat formats render ChessGPT's chat template with a system message
    around the format's prompt template, and the model sees exactly that text
    with every Outlines version. The ``TextPromptFormat.CHAT`` template is
    itself a ChessGPT chat turn, so its prompt contains the system text twice::

        A friendly, helpful chat between some humans.<|endoftext|>Human 0: A friendly, helpful chat between some humans.<|endoftext|>Human 0: {prompt}

        PGN: {pgn}

        Just write your move.<|endoftext|>Human 1:

    This worked better than applying the template once, which
    ``TextPromptFormat.CHAT_ORIGINAL`` does::

        A friendly, helpful chat between some humans.<|endoftext|>Human 0: {prompt}

        PGN: {pgn}

        Just write your move.<|endoftext|>Human 1:
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = torch.bfloat16,
        temperature: float = 0.0,
        prompt_template: str | None = None,
        chat_template: str | None = _FORMAT_DEFAULT,
        input_config: PipelineInputConfig | None = None,
        constraint: tp.Literal["regex", "choice"] | None = None,
        pipeline_config: TextGenerationPipelineConfig | None = None,
        **kwargs: tp.Any,
    ):
        if pipeline_config is None:
            pipeline_config = TextGenerationPipelineConfig()
        if constraint is not None:
            pipeline_config = dataclasses.replace(
                pipeline_config, constraint=constraint
            )
        # get_generator_for_model, called by the base class, reads the config.
        self.pipeline_config = pipeline_config
        if prompt_template is None:
            prompt_template = pipeline_config.default_prompt_template
        if chat_template is _FORMAT_DEFAULT:
            chat_template = pipeline_config.default_chat_template
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            temperature=temperature,
            prompt_template=prompt_template,
            chat_template=chat_template,
            input_config=input_config or pipeline_config.input_config,
            **kwargs,
        )

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return TextGenerationPipelineConfig().input_config

    @property
    def constraint(self) -> str:
        return self.pipeline_config.constraint

    def get_generator_for_model(self, model: outlines.Model, *args, **kwargs):
        if self.constraint == "choice":
            return None
        return outlines.Generator(
            model,
            outlines.types.Regex(self.pipeline_config.move_regex),
        )

    def legal_move_choices(
        self,
        history: board_history.BoardHistory,
        legal_moves: list[chess.Move],
    ) -> list[str]:
        """Return the outputs ``constraint="choice"`` allows for a position."""
        if self.pipeline_config.prompt_format in _CHAT_FORMATS:
            return numbered_move_choices(history, legal_moves)
        return pgn_continuation_move_choices(history, legal_moves)

    def forward_batch(
        self,
        consolidated_prompts: list[str],
        histories: list[board_history.BoardHistory],
        legal_moves: list[list[chess.Move]],
    ):
        if self.constraint == "regex":
            return super().forward_batch(
                consolidated_prompts,
                histories,
                legal_moves,
            )

        results = []
        for prompt, history, moves in zip(consolidated_prompts, histories, legal_moves):
            generator = outlines.Generator(
                self.model,
                outlines.types.Choice(self.legal_move_choices(history, moves)),
            )
            results.append(generator(prompt, temperature=self.temperature))
        return results

    def make_prediction(
        self,
        pgn: str | list[str],
        *inputs: tp.Any,
        prompt: str | list[str] | None = None,
        time_control: str | list[str] | None = None,
        white_elo: int | list[int] | None = None,
        black_elo: int | list[int] | None = None,
        should_shuffle_legal_moves: bool = False,
        return_histories: bool = False,
        tqdm_leave: bool = False,
        batch_size: int = 16,
        persistent_workers: bool = False,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        """Predict a move for each PGN.

        The other inputs depend on the prompt format: ``prompt`` for the chat
        and ``PGN_HEADER`` formats; ``time_control``, ``white_elo`` and
        ``black_elo`` for the others. They may also be passed positionally
        after the PGNs, in that order.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        values = bind_positional_inputs(
            self.pipeline_config.input_names,
            inputs,
            {
                "prompts": prompt,
                "time_controls": time_control,
                "white_elos": white_elo,
                "black_elos": black_elo,
            },
        )
        provided = {name: value for name, value in values.items() if value is not None}
        self.input_config.validate(
            pgns=pgn, **provided, **extract_domain_inputs(kwargs)
        )
        item_count = 1 if isinstance(pgn, str) else len(pgn)
        resolved_workers = resolve_num_workers(
            item_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )
        prompt_format = self.pipeline_config.prompt_format
        if should_shuffle_legal_moves and self.pipeline_config.input_names != (
            _PROMPT_INPUTS
        ):
            raise ValueError(
                "should_shuffle_legal_moves is only supported by formats "
                "that take a prompt"
            )
        common = {
            "num_workers": resolved_workers,
            "parallel_threshold": parallel_threshold,
        }
        if prompt_format == TextPromptFormat.GENERAL_POLICY:
            dataset = _GeneralPolicyDataset(
                pgn=pgn,
                time_control=values["time_controls"],
                white_elo=values["white_elos"],
                black_elo=values["black_elos"],
                prompt_template=self.prompt_template,
                **common,
            )
        elif prompt_format == TextPromptFormat.PGN_HEADER_WITH_TIME_CONTROL_AND_ELO:
            dataset = _TimeControlAndEloDataset(
                pgns=pgn,
                time_control=values["time_controls"],
                white_elo=values["white_elos"],
                black_elo=values["black_elos"],
                **common,
            )
        else:
            dataset = _PromptDataset(
                pgns=pgn,
                prompts=values["prompts"],
                should_shuffle_legal_moves=should_shuffle_legal_moves,
                prompt_template=self.prompt_template,
                **common,
            )
        return self._make_predictions_from_dataset(
            dataset,
            return_histories=return_histories,
            tqdm_leave=tqdm_leave,
            batch_size=batch_size,
            persistent_workers=persistent_workers,
            num_workers=resolved_workers,
            parallel_threshold=parallel_threshold,
        )

    @classmethod
    def post_process_success(
        cls,
        response: str,
        history: board_history.BoardHistory,
        legal_moves: list[chess.Move],
        return_histories: bool = False,
    ):
        try:
            chess_response = ChessResponse(
                moves=[ChessMove(move=response.strip(), probability=1.0)]
            )
        except AttributeError:
            logging.error("Could not parse the response %s", response)
            raise
        best_move_san, best_move_uci, preds_list = (
            chess_response.to_best_moves_and_preds_list(
                history.active_board, rescale_excess=True, fill_missing=True
            )
        )
        if return_histories:
            best_move = chess.Move.from_uci(best_move_uci)
            history.append(best_move)
        return LLMChessMovePrediction(
            board_history=history if return_histories else None,
            best_move_san=best_move_san,
            best_move_uci=best_move_uci,
            preds_list=preds_list,
            raw_response=response,
        )


__all__ = [
    "AsyncOpenAIProbabilityPipeline",
    "CHESSGPT_SYSTEM_PROMPT",
    "DEFAULT_BASIC_PROMPT_TEMPLATE",
    "DEFAULT_OUTLINES_PROMPT_TEMPLATE",
    "DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO",
    "END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE",
    "NEWLINE_CHESSGPT_TEMPLATE",
    "NUMBERED_MOVE_REGEX",
    "OpenAIProbabilityPipeline",
    "OutlinesProbabilityPipeline",
    "PAPER_CHESSGPT_TEMPLATE",
    "PGN_CONTINUATION_MOVE_REGEX",
    "PGN_HEADER_PROMPT_TEMPLATE",
    "TextGenerationPipeline",
    "TextGenerationPipelineConfig",
    "TextPromptFormat",
    "TransformersProbabilityPipeline",
    "WATERHORSE_CHESSGPT_GENERAL_POLICY_PROMPT_TEMPLATE",
    "WATERHORSE_CHESSGPT_PROMPT_TEMPLATE",
    "WATERHORSE_CHESSGPT_PROMPT_TEMPLATE_NEWLINE",
    "WATERHORSE_CHESSGPT_PROMPT_TEMPLATE_RAW",
    "numbered_move_choices",
    "pgn_continuation_move_choices",
    "render_chat_prompt_template",
]
