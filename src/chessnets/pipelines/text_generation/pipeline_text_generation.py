"""Transformers, Outlines, and OpenAI-backed chess text pipelines."""

from __future__ import annotations

import asyncio
import importlib.metadata
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

from chessnets._optional import import_optional, optional_module
from chessnets.utils import board_history

from ..base import Pipeline, PipelineInputConfig, extract_domain_inputs
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

jinja2 = optional_module("jinja2", "text-generation")
openai = optional_module("openai", "text-generation")
outlines = optional_module("outlines", "text-generation")
packaging_version = optional_module("packaging.version", "text-generation")
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


class OutlinesDataset(torch.utils.data.Dataset):
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


class ChessGPTGeneralPolicyDataset(torch.utils.data.Dataset):
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


class OutlinesProbabilityTextPipeline(Pipeline):
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
        dataset = OutlinesDataset(
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


class OpenAIProbabilityTextPipeline(OutlinesProbabilityTextPipeline):
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


class AsyncOpenAIProbabilityTextPipeline(OutlinesProbabilityTextPipeline):
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


PAPER_CHESSGPT_TEMPLATE = "{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim + '<|endoftext|>' if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}\n{%- for message in messages -%}\n{%- if message['role'] == 'user' -%}\nHuman 0: {{ message['content']|trim -}}\n{%- if not loop.last -%}\n<|endoftext|>\n{%- endif -%}\n{%- elif message['role'] == 'assistant' -%}\nHuman 1: {{ message['content']|trim -}}\n{%- if not loop.last -%}\n<|endoftext|>\n{%- endif -%}\n{%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}\n<|endoftext|>Human 1: {% endif -%}\n"
NEWLINE_CHESSGPT_TEMPLATE = "{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim + '<|endoftext|>' if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}\n{%- for message in messages -%}\n{%- if message['role'] == 'user' -%}\nHuman 0: {{ '\n'~message['content']|trim -}}\n{%- if not loop.last -%}\n<|endoftext|>\n{%- endif -%}\n{%- elif message['role'] == 'assistant' -%}\nHuman 1: {{ '\n'~message['content']|trim -}}\n{%- if not loop.last -%}\n<|endoftext|>\n{%- endif -%}\n{%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt and messages[-1]['role'] != 'assistant' -%}\n<|endoftext|>Human 1: {% endif -%}\n"


class HFTransformersProbabilityTextPipeline(OutlinesProbabilityTextPipeline):
    auto_batch_sizes = {"cpu": 4, "cuda": 8}

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.0,
        prompt_template: str = DEFAULT_OUTLINES_PROMPT_TEMPLATE,
        chat_template: str | None = PAPER_CHESSGPT_TEMPLATE,
        input_config: PipelineInputConfig | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        model_kwargs: dict[str, tp.Any] | None = None,
        tokenizer_kwargs: dict[str, tp.Any] | None = None,
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
        # Waterhorse/chessgpt-chat-v1 chat template:
        tokenizer.chat_template = chat_template
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
    ):
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
        try:
            import flash_attn  # noqa: F401 -- probe optional dependency availability

            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                attn_implementation = "flash_attention_2"
            else:
                attn_implementation = "sdpa"
        except ImportError:
            attn_implementation = "sdpa"
        resolved_model_kwargs = {**common_kwargs, **(model_kwargs or {})}
        resolved_model_kwargs.setdefault("dtype", dtype)
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


class HFTransformersSimpleTextPipeline(HFTransformersProbabilityTextPipeline):
    """A simpler pipeline that just asks the model to write a move, without asking for probabilities.

    This is useful for models that don't do well with the more complex prompt or output format, such as Waterhorse's ChessGPT models.

    It's also better for efficient inference.

    TODO (s8siu): Maybe refactor code to just use outlines.type.Chat?
    Note: When using outlines < 1.2.10 and the tokenizer's chat template has not been set, the following call returns ['1. e4'] instead of ['2. c3'] as expected, and a lot of other weird outputs.
        Issue: https://github.com/dottxt-ai/outlines/issues/1784
        PR: https://github.com/dottxt-ai/outlines/pull/1789

        >>> pipeline.generator.batch(['A friendly, helpful chat between some humans. Human 0: In the Sicilian Defense: Alapin Variation, the player as White takes a strategic approach by challenging Black’s setup early with 1. e4 c5, thus steering the game into less trodden paths and potentially catching their opponent off-guard.\n\nPGN: 1. e4 c5\n\nJust write your move.'])
        ['2. c3']

        Honestly, the actual prompt might be prompt_template(prompt_template(prompt, add_generation_prompt=False), add_generation_prompt=True), but it seems to work better than just applying the template once
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.0,
        prompt_template: str = WATERHORSE_CHESSGPT_PROMPT_TEMPLATE,
        chat_template: str | None = PAPER_CHESSGPT_TEMPLATE,
        input_config: PipelineInputConfig | None = None,
        constraint: str = "regex",
        **kwargs: tp.Any,
    ):
        if constraint not in {"regex", "choice"}:
            raise ValueError("constraint must be either 'regex' or 'choice'")
        self.constraint = constraint
        # Import first so a missing extra reports how to install it.
        import_optional("outlines", "text-generation")
        outlines_version = importlib.metadata.version("outlines")
        if chat_template is not None and packaging_version.parse(
            outlines_version
        ) < packaging_version.parse("1.2.10"):
            prompt_template = jinja2.Template(chat_template).render(
                messages=[
                    {
                        "role": "system",
                        "content": "A friendly, helpful chat between some humans.",
                    },
                    {"role": "user", "content": prompt_template},
                ],
                add_generation_prompt=True,
            )
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            temperature=temperature,
            prompt_template=prompt_template,
            chat_template=chat_template,
            input_config=input_config,
            **kwargs,
        )

    def get_generator_for_model(self, model: outlines.Model, *args, **kwargs):
        if self.constraint == "choice":
            return None
        return outlines.Generator(
            model,
            outlines.types.Regex(NUMBERED_MOVE_REGEX),
        )

    @staticmethod
    def legal_move_choices(
        history: board_history.BoardHistory,
        legal_moves: list[chess.Move],
    ) -> list[str]:
        """Return standalone, numbered legal moves accepted by this profile."""
        board = history.active_board
        dots = "." if board.turn == chess.WHITE else "..."
        prefix = f"{board.fullmove_number}{dots} "
        return [prefix + board.san(move) for move in legal_moves]

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
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON object")
            return cls.post_process_failure(
                history, legal_moves, return_histories=return_histories
            )
        except AttributeError as e:
            logging.error("Could not parse the response %s", response)
            raise e
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


class HFTransformersSimpleTextPipelineOriginal(HFTransformersSimpleTextPipeline):
    """A simpler pipeline that just asks the model to write a move, without asking for probabilities.

    This is useful for models that don't do well with the more complex prompt or output format, such as Waterhorse's ChessGPT models.

    It's also better for efficient inference.

    Unlike HFTransformersSimpleTextPipeline, this class just applies the chat template once.
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.0,
        prompt_template: str = WATERHORSE_CHESSGPT_PROMPT_TEMPLATE_RAW,
        chat_template: str | None = PAPER_CHESSGPT_TEMPLATE,
        input_config: PipelineInputConfig | None = None,
        **kwargs: tp.Any,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            temperature=temperature,
            prompt_template=prompt_template,
            chat_template=chat_template,
            input_config=input_config,
            **kwargs,
        )


class HFTransformersSimpleTextPipelineGeneralPolicy(HFTransformersSimpleTextPipeline):
    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.0,
        prompt_template: str = WATERHORSE_CHESSGPT_GENERAL_POLICY_PROMPT_TEMPLATE,
        chat_template: str | None = PAPER_CHESSGPT_TEMPLATE,
        **kwargs: tp.Any,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            temperature=temperature,
            prompt_template=prompt_template,
            chat_template=chat_template,
            **kwargs,
        )

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(
            required=("pgns", "time_controls", "white_elos", "black_elos")
        )

    def make_prediction(
        self,
        pgn: str | list[str],
        time_control: str | list[str],
        white_elo: int | list[int],
        black_elo: int | list[int],
        *args,
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
            pgns=pgn,
            time_controls=time_control,
            white_elos=white_elo,
            black_elos=black_elo,
            **extract_domain_inputs(kwargs),
        )
        item_count = 1 if isinstance(pgn, str) else len(pgn)
        resolved_workers = resolve_num_workers(
            item_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )
        dataset = ChessGPTGeneralPolicyDataset(
            pgn=pgn,
            time_control=time_control,
            white_elo=white_elo,
            black_elo=black_elo,
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


PGN_HEADER_PROMPT_TEMPLATE = "{prompt}\n\n{pgn}"
END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE = "{prompt}<|endoftext|>{pgn}"


class HFTransformersSimpleTextPipelinePgnHeader(HFTransformersSimpleTextPipeline):
    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        temperature: float = 0.0,
        prompt_template: str = PGN_HEADER_PROMPT_TEMPLATE,
        chat_template: str | None = None,
        input_config: PipelineInputConfig | None = None,
        **kwargs: tp.Any,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            temperature=temperature,
            prompt_template=prompt_template,
            chat_template=chat_template,
            input_config=input_config,
            **kwargs,
        )

    def get_generator_for_model(self, model: outlines.Model, *args, **kwargs):
        if self.constraint == "choice":
            return None
        return outlines.Generator(
            model,
            outlines.types.Regex(PGN_CONTINUATION_MOVE_REGEX),
        )

    @staticmethod
    def legal_move_choices(
        history: board_history.BoardHistory,
        legal_moves: list[chess.Move],
    ) -> list[str]:
        """Return the legal PGN-continuation variants allowed by the regex."""
        board = history.active_board
        dots = "." if board.turn == chess.WHITE else "..."
        prefix = f"{board.fullmove_number}{dots} "
        choices = []
        for move in legal_moves:
            san = board.san(move)
            choices.extend((san, f" {san}", prefix + san, " " + prefix + san))
        return choices


class LocalHFTransformersSimpleTextPipelinePgnHeader(
    HFTransformersSimpleTextPipelinePgnHeader
):
    def load_model_and_tokenizer(
        self,
        model_id: str,
        device: str | torch.device = "cpu",
        dtype: str | torch.dtype | None = None,
        **kwargs: tp.Any,
    ):
        tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", None) or {}
        model_kwargs = kwargs.pop("model_kwargs", None) or {}
        common_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"revision", "cache_dir", "token", "local_files_only"}
        }
        resolved_tokenizer_kwargs = {**common_kwargs, **tokenizer_kwargs}
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, **resolved_tokenizer_kwargs
        )
        resolved_model_kwargs = {**common_kwargs, **model_kwargs}
        resolved_model_kwargs.setdefault("dtype", dtype)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, **resolved_model_kwargs
        ).to(device=device)
        return model, tokenizer


DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO = """[White "???"]\n[Black "???"]\n[WhiteElo "{white_elo}"]\n[BlackElo "{black_elo}"]\n[TimeControl "{time_control}"]\n\n"""


class ChessGPTBaseDatasetWithTimeControlAndElo(torch.utils.data.Dataset):
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


class HFTransformersSimpleTextPipelineCompletionWithTimeControlAndElo(
    HFTransformersSimpleTextPipelinePgnHeader
):
    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(
            required=("pgns", "time_controls", "white_elos", "black_elos")
        )

    def make_prediction(
        self,
        pgn: str | list[str],
        time_control: list[str] | str,
        white_elo: list[int] | int,
        black_elo: list[int] | int,
        *args,
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
            pgns=pgn,
            time_controls=time_control,
            white_elos=white_elo,
            black_elos=black_elo,
            **extract_domain_inputs(kwargs),
        )
        item_count = 1 if isinstance(pgn, str) else len(pgn)
        resolved_workers = resolve_num_workers(
            item_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )
        dataset = ChessGPTBaseDatasetWithTimeControlAndElo(
            pgns=pgn,
            time_control=time_control,
            white_elo=white_elo,
            black_elo=black_elo,
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


__all__ = [
    "ChessGPTGeneralPolicyDataset",
    "ChessGPTBaseDatasetWithTimeControlAndElo",
    "DEFAULT_BASIC_PROMPT_TEMPLATE",
    "DEFAULT_OUTLINES_PROMPT_TEMPLATE",
    "DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO",
    "END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE",
    "AsyncOpenAIProbabilityTextPipeline",
    "HFTransformersProbabilityTextPipeline",
    "HFTransformersSimpleTextPipeline",
    "HFTransformersSimpleTextPipelineGeneralPolicy",
    "HFTransformersSimpleTextPipelineOriginal",
    "HFTransformersSimpleTextPipelinePgnHeader",
    "HFTransformersSimpleTextPipelineCompletionWithTimeControlAndElo",
    "LocalHFTransformersSimpleTextPipelinePgnHeader",
    "NEWLINE_CHESSGPT_TEMPLATE",
    "OpenAIProbabilityTextPipeline",
    "OutlinesDataset",
    "OutlinesProbabilityTextPipeline",
    "PAPER_CHESSGPT_TEMPLATE",
    "PGN_HEADER_PROMPT_TEMPLATE",
    "WATERHORSE_CHESSGPT_GENERAL_POLICY_PROMPT_TEMPLATE",
    "WATERHORSE_CHESSGPT_PROMPT_TEMPLATE",
    "WATERHORSE_CHESSGPT_PROMPT_TEMPLATE_NEWLINE",
    "WATERHORSE_CHESSGPT_PROMPT_TEMPLATE_RAW",
]
