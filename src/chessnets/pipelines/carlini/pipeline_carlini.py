"""Carlini and Paleka OpenAI Instruct Chess pipeline.

Prompt format from N. Carlini, "Playing chess with large language models" (2023),
https://nicholas.carlini.com/writing/2023/chess-llm.html, and Carlini & Paleka's
chess-llm code. Reimplemented for ChessNets; no upstream source code is included.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import logging
import pathlib
import random
import typing as tp
import warnings

import chess
import torch
import torch.utils.data

from chessnets._optional import optional_module
from chessnets.utils import board_history

from ..base import Pipeline, PipelineInputConfig, extract_domain_inputs
from ..outputs import ChessMovePrediction, ChessMovePredictionDataset
from ..pipeline_utils import (
    DEFAULT_PARALLEL_THRESHOLD,
    adaptive_process_map,
    asyncio_run,
    iter_dataset_batches,
    progress,
    resolve_num_workers,
)

openai = optional_module("openai", "text-generation")

DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO = """[White \"???\"]
[Black \"???\"]
[WhiteElo \"{white_elo}\"]
[BlackElo \"{black_elo}\"]
[TimeControl \"{time_control}\"]

"""


def _single_item_collate(batch):
    """Preserve the old ``batch_size=None`` DataLoader contract."""
    return batch[0]


def _pgn_movetext(history: board_history.BoardHistory, warn_on_game_over: bool) -> str:
    """Render the common movetext portion used by every Carlini prompt."""
    board = history.active_board
    pgn = str(chess.pgn.Game().from_board(board))

    if board.outcome() is None:
        pgn = pgn[:-1].strip()
    else:
        logging.info("Game is over; no moves valid")
        error_message = f"Game is over; no moves valid. FEN: {board.fen()}  PGN: {pgn}"
        if warn_on_game_over:
            warnings.warn(error_message, stacklevel=2)
        else:
            raise ValueError(error_message)
        pgn = pgn.rsplit(" ", 1)[0].strip()

    if board.turn == chess.WHITE:
        suffix = (
            "\n\n1." if board.fullmove_number == 1 else f" {board.fullmove_number}."
        )
        pgn += suffix

    return pgn.split("\n\n", 1)[1]


DEFAULT_CARLINI_PGN_HEADER_CODE = """[White "Magnus Carlsen"]\n[Black "Garry Kasparov"]\n[WhiteElo "2900"]\n[BlackElo "2800"]\n\n"""
DEFAULT_CARLINI_PGN_HEADER_BLOG = """[White "Garry Kasparov"]\n[Black "Magnus Carlsen"]\n[Result "1/2-1/2"]\n[WhiteElo "2900"]\n[BlackElo "2800"]\n\n"""
DEFAULT_CARLINI_NUM_LOOKAHEAD_TOKENS = 9


class CarliniInputMode(str, enum.Enum):
    """Input contract for the Carlini pipeline."""

    DEFAULT = "default"
    PROMPT = "prompt"
    TIME_CONTROL_AND_ELO = "time_control_and_elo"


@dataclasses.dataclass(frozen=True)
class CarliniPipelineConfig:
    """Declarative options shared by the Carlini pipeline variants."""

    input_mode: CarliniInputMode = CarliniInputMode.DEFAULT
    num_lookahead_tokens: int = DEFAULT_CARLINI_NUM_LOOKAHEAD_TOKENS
    warn_on_game_over: bool = True
    pgn_header_override: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_mode, CarliniInputMode):
            try:
                object.__setattr__(
                    self, "input_mode", CarliniInputMode(self.input_mode)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported Carlini input mode: {self.input_mode!r}"
                ) from exc
        if self.num_lookahead_tokens < DEFAULT_CARLINI_NUM_LOOKAHEAD_TOKENS:
            raise ValueError("num_lookahead_tokens must be at least 9")

    @property
    def input_config(self) -> PipelineInputConfig:
        if self.input_mode == CarliniInputMode.DEFAULT:
            return PipelineInputConfig.from_inputs(required=("pgns",))
        if self.input_mode == CarliniInputMode.PROMPT:
            return PipelineInputConfig.from_inputs(required=("pgns", "prompts"))
        return PipelineInputConfig.from_inputs(
            required=("pgns", "time_controls", "white_elos", "black_elos")
        )


class CarliniOpenAIInstructChessDataset(torch.utils.data.Dataset):
    """Dataset for Carlini and Paleka's OpenAI Instruct Chess Pipeline."""

    def __init__(
        self,
        pgns: list[str] | str,
        warn_on_game_over: bool = True,
        pgn_header_override: str | None = None,
        *,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
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
            self.pgn_header = DEFAULT_CARLINI_PGN_HEADER_CODE
        self.prompts: list[str] = adaptive_process_map(
            self.create_prompt,
            self.histories,
            [warn_on_game_over] * len(self.histories),
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
        history: board_history.BoardHistory,
        warn_on_game_over: bool,
        pgn_header: str = DEFAULT_CARLINI_PGN_HEADER_CODE,
    ) -> str:
        return pgn_header + _pgn_movetext(history, warn_on_game_over)

    def __getitem__(self, idx: int):
        prompt = self.prompts[idx]
        history = self.histories[idx]
        return prompt, history

    @staticmethod
    def collate_fn(batch):
        prompts, histories = zip(*batch)
        return list(prompts), list(histories)

    @staticmethod
    def async_collate_fn(batch):
        prompts, histories = zip(*batch)
        seen_fens = set()
        first_fen_indices = []
        seen_fen_indices = []
        for i, history in enumerate(histories):
            fen = history.active_board.fen()
            if fen not in seen_fens:
                seen_fens.add(fen)
                first_fen_indices.append(i)
            else:
                seen_fen_indices.append(i)
        prompts = [prompts[i] for i in first_fen_indices]
        return prompts, list(histories), first_fen_indices, seen_fen_indices


class CarliniOpenAIInstructChessDatasetWithPrompt(CarliniOpenAIInstructChessDataset):
    """Dataset for Carlini and Paleka's OpenAI Instruct Chess Pipeline, except this one also inserts the prompt after the PGN header."""

    def __init__(
        self,
        pgns: list[str] | str,
        prompts: list[str] | str,
        warn_on_game_over: bool = True,
        pgn_header_override: str | None = None,
        *,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
        if isinstance(prompts, str):
            prompts = [prompts]
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
            self.pgn_header = DEFAULT_CARLINI_PGN_HEADER_CODE
        self.prompts: list[str] = adaptive_process_map(
            self.create_prompt,
            self.histories,
            prompts,
            [warn_on_game_over] * len(self.histories),
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
        history: board_history.BoardHistory,
        prompt: str,
        warn_on_game_over: bool,
        pgn_header: str = DEFAULT_CARLINI_PGN_HEADER_CODE,
    ) -> str:
        return pgn_header + f"{prompt}\n\n" + _pgn_movetext(history, warn_on_game_over)


class CarliniOpenAIInstructChessDatasetWithTimeControlAndElo(
    CarliniOpenAIInstructChessDataset
):
    """Dataset for Carlini and Paleka's OpenAI Instruct Chess Pipeline,
    except this one uses the actual time control and Elo rating information."""

    def __init__(
        self,
        pgns: list[str] | str,
        time_control: list[str] | str,
        white_elo: list[int] | int,
        black_elo: list[int] | int,
        warn_on_game_over: bool = True,
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
            self.histories,
            time_control,
            white_elo,
            black_elo,
            [warn_on_game_over] * len(self.histories),
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
        history: board_history.BoardHistory,
        time_control: str,
        white_elo: int,
        black_elo: int,
        warn_on_game_over: bool,
        pgn_header: str = DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO,
    ) -> str:
        return pgn_header.format(
            time_control=time_control,
            white_elo=white_elo,
            black_elo=black_elo,
        ) + _pgn_movetext(history, warn_on_game_over)


class CarliniOpenAIInstructChessPipeline(Pipeline):
    """Carlini and Paleka's OpenAI Instruct Chess Pipeline."""

    def __init__(
        self,
        model_id: str,
        num_lookahead_tokens: int | None = None,
        cache_path_template: str = "~/.cache/openai_instruct_chess/{model_id}/cache.json",
        warn_on_game_over: bool | None = None,
        pgn_header_override: str | None = None,
        client_factory: tp.Any = None,
        input_config: PipelineInputConfig | None = None,
        pipeline_config: CarliniPipelineConfig | None = None,
    ):
        if pipeline_config is None:
            # Infer the compatibility mode used by the legacy subclasses from
            # their overridden input contract.
            legacy_config = self.default_input_config()
            required = legacy_config.required_inputs
            if "prompts" in required:
                mode = CarliniInputMode.PROMPT
            elif "time_controls" in required or "white_elos" in required:
                mode = CarliniInputMode.TIME_CONTROL_AND_ELO
            else:
                mode = CarliniInputMode.DEFAULT
            pipeline_config = CarliniPipelineConfig(input_mode=mode)
        expected_input_config = pipeline_config.input_config
        if input_config is not None:
            if input_config.required_inputs != expected_input_config.required_inputs:
                raise ValueError(
                    "The supplied input_config is incompatible with the selected "
                    f"Carlini input mode ({pipeline_config.input_mode.value})"
                )
            if (
                input_config.allowed_inputs is not None
                and input_config.allowed_inputs != expected_input_config.allowed_inputs
            ):
                raise ValueError(
                    "The supplied input_config accepts inputs incompatible with "
                    f"Carlini input mode ({pipeline_config.input_mode.value})"
                )
        super().__init__(input_config=expected_input_config)
        self.pipeline_config = pipeline_config
        self.model_id = model_id
        self.num_lookahead_tokens = (
            pipeline_config.num_lookahead_tokens
            if num_lookahead_tokens is None
            else num_lookahead_tokens
        )
        if self.num_lookahead_tokens is not None:
            assert self.num_lookahead_tokens >= DEFAULT_CARLINI_NUM_LOOKAHEAD_TOKENS, (
                "A single move might take as many as 9 tokens (3 for the number + 6 for, e.g., 'N3xg5+')."
            )

        self.cache_path = (
            pathlib.Path(cache_path_template.format(model_id=self.model_id))
            .expanduser()
            .resolve()
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.cache_path.exists():
            with self.cache_path.open() as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        logging.info(f"Loading cache with {len(self.cache)} entries")

        if client_factory is None:
            client_factory = openai.OpenAI
        self.client = client_factory(max_retries=MAX_RETRIES)
        self.warn_on_game_over = (
            pipeline_config.warn_on_game_over
            if warn_on_game_over is None
            else warn_on_game_over
        )
        self.pgn_header_override = (
            pipeline_config.pgn_header_override
            if pgn_header_override is None
            else pgn_header_override
        )

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return CarliniPipelineConfig().input_config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, **kwargs: tp.Any
    ) -> tp.Self:
        """Select the hosted OpenAI completion model used by Carlini."""
        return cls(model_id=pretrained_model_name_or_path, **kwargs)

    def try_moves(self, board: chess.Board, next_text: str) -> list[str]:
        board = board.copy()
        moves = next_text.split()
        ok_moves = []
        for move in moves:
            if "." in move:
                continue
            try:
                board.push_san(move)
                ok_moves.append(move)
            except ValueError:  # python-chess's invalid/illegal/ambiguous SAN errors
                break

        return ok_moves

    def _resolve_num_tokens(self, num_tokens: int | None) -> int:
        if num_tokens is None:
            num_tokens = self.num_lookahead_tokens
        if num_tokens is None:
            num_tokens = DEFAULT_CARLINI_NUM_LOOKAHEAD_TOKENS
        assert num_tokens >= DEFAULT_CARLINI_NUM_LOOKAHEAD_TOKENS, (
            "A single move might take as many as 9 tokens (3 for the number + 6 for, e.g., 'N3xg5+')."
        )
        return num_tokens

    def _make_dataset(
        self,
        pgns: list[str] | str,
        *,
        prompts: list[str] | str | None = None,
        time_controls: list[str] | str | None = None,
        white_elos: list[int] | int | None = None,
        black_elos: list[int] | int | None = None,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    ) -> CarliniOpenAIInstructChessDataset:
        """Build the dataset selected by the declarative input mode."""
        common = {
            "pgns": pgns,
            "warn_on_game_over": self.warn_on_game_over,
            "pgn_header_override": self.pgn_header_override,
            "num_workers": num_workers,
            "parallel_threshold": parallel_threshold,
        }
        config = getattr(self, "pipeline_config", CarliniPipelineConfig())
        mode = config.input_mode
        if mode == CarliniInputMode.PROMPT:
            return CarliniOpenAIInstructChessDatasetWithPrompt(
                prompts=prompts, **common
            )
        if mode == CarliniInputMode.TIME_CONTROL_AND_ELO:
            return CarliniOpenAIInstructChessDatasetWithTimeControlAndElo(
                time_control=time_controls,
                white_elo=white_elos,
                black_elo=black_elos,
                **common,
            )
        return CarliniOpenAIInstructChessDataset(**common)

    def make_prediction(
        self,
        pgns: list[str] | str,
        num_tokens: int | None = None,
        *,
        prompts: list[str] | str | None = None,
        time_controls: list[str] | str | None = None,
        white_elos: list[int] | int | None = None,
        black_elos: list[int] | int | None = None,
        verbose: bool = False,
        tqdm_leave: bool = False,
        return_histories: bool = False,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        if not hasattr(self, "input_config"):
            self.input_config = self.default_input_config()
        pipeline_config = getattr(self, "pipeline_config", None)
        if pipeline_config is None:
            pipeline_config = CarliniPipelineConfig()
        num_tokens = self._resolve_num_tokens(num_tokens)
        self.input_config.validate(
            pgns=pgns,
            prompts=prompts,
            time_controls=time_controls,
            white_elos=white_elos,
            black_elos=black_elos,
            **extract_domain_inputs(kwargs),
        )
        item_count = 1 if isinstance(pgns, str) else len(pgns)
        resolved_workers = resolve_num_workers(
            item_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )

        dataset = self._make_dataset(
            pgns,
            prompts=prompts,
            time_controls=time_controls,
            white_elos=white_elos,
            black_elos=black_elos,
            num_workers=resolved_workers,
            parallel_threshold=parallel_threshold,
        )
        raw_outputs = []
        for prompt, history in progress(
            iter_dataset_batches(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=_single_item_collate,
                # Prompt/history preparation is eager; workers would only
                # fetch already-materialized pairs here.
                num_workers=0,
            ),
            desc="Making predictions",
            total=len(dataset),
            leave=tqdm_leave,
        ):
            board = history.active_board
            if board.fen() in self.cache:
                out = self.cache[board.fen()]
                if verbose:
                    if board.ply() > 0:
                        logging.debug(
                            f"LLM played a move already in my cache. Returning {out}.",
                        )
                if return_histories:
                    raw_outputs.append((history, out))
                else:
                    raw_outputs.append((history.active_board, out))
                continue

            if verbose:
                logging.debug(
                    f"Querying {self.model_id} with ... {prompt.split(']')[-1][-90:]}",
                )

            next_text = self.forward(prompt, num_tokens)
            if next_text[:2] == "-O":
                next_text = self.forward(prompt + " ", num_tokens)

            if verbose:
                logging.debug(f"Received reply of '{next_text}'")

            next_moves = self.try_moves(board, next_text)

            if len(next_moves) == 0:
                logging.debug("Tried to make an invalid move.")
                if return_histories:
                    raw_outputs.append((history, None))
                else:
                    raw_outputs.append((history.active_board, None))
                continue

            if verbose:
                logging.debug(f"Received reply and making move {next_moves[0]}.")

            new_board = board.copy()
            for move in next_moves:
                self.cache[new_board.fen()] = move
                new_board.push_san(move)

            if return_histories:
                raw_outputs.append((history, next_moves[0]))
            else:
                raw_outputs.append((history.active_board, next_moves[0]))
        with self.cache_path.open("w") as f:
            json.dump(self.cache, f)

        if raw_outputs:
            predictions = adaptive_process_map(
                self.post_process_predictions,
                *zip(*raw_outputs),
                parallel_threshold=parallel_threshold,
                max_workers=resolved_workers,
                desc="Collecting predictions",
                leave=tqdm_leave,
                chunksize=64,
            )
        else:
            predictions = []
        return ChessMovePredictionDataset[ChessMovePrediction](predictions=predictions)

    @staticmethod
    def post_process_predictions(
        board_or_history: chess.Board | board_history.BoardHistory,
        pred_move_san: str | None,
    ) -> ChessMovePrediction:
        if isinstance(board_or_history, board_history.BoardHistory):
            board = board_or_history.active_board
        else:
            board = board_or_history

        board_history_ = (
            board_or_history
            if isinstance(board_or_history, board_history.BoardHistory)
            else None
        )
        if pred_move_san is None:
            # If there's an error, just pick a random legal move
            legal_moves = list(board.legal_moves)
            best_move = random.choice(legal_moves)
            best_move_san = board.san(best_move)
            best_move_uci = str(best_move)
            preds_list = {m: 1 / len(legal_moves) for m in legal_moves}
            return ChessMovePrediction(
                board_history=board_history_,
                best_move_san=best_move_san,
                best_move_uci=best_move_uci,
                preds_list=preds_list,
            )
        best_move = board.parse_san(pred_move_san)
        best_move_uci = str(best_move)
        preds_list = {best_move: 1.0}
        return ChessMovePrediction(
            board_history=board_history_,
            best_move_san=pred_move_san,
            best_move_uci=best_move_uci,
            preds_list=preds_list,
        )

    def forward(self, prompt: str, num_tokens: int) -> str | list[str]:
        try:
            response = self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                temperature=0.0,
                max_tokens=num_tokens,
            )
        except openai.BadRequestError as e:
            logging.error(f"Error from OpenAI API: {e}")
            logging.error(f"Prompt that caused the error: {prompt}")
            raise e

        messages = [choice.text for choice in response.choices]

        if len(messages) == 1:
            return messages[0]
        else:
            return [message for message in messages]


class CarliniOpenAIInstructChessPipelineWithPrompt(CarliniOpenAIInstructChessPipeline):
    """Carlini and Paleka's OpenAI Instruct Chess Pipeline, except this one also inserts the prompt after the PGN header."""

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return CarliniPipelineConfig(input_mode=CarliniInputMode.PROMPT).input_config

    def make_prediction(
        self,
        pgns: list[str] | str,
        prompts: list[str] | str,
        num_tokens: int | None = None,
        verbose: bool = False,
        tqdm_leave: bool = False,
        return_histories: bool = False,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        return super().make_prediction(
            pgns,
            num_tokens,
            prompts=prompts,
            verbose=verbose,
            tqdm_leave=tqdm_leave,
            return_histories=return_histories,
            num_workers=num_workers,
            parallel_threshold=parallel_threshold,
            **kwargs,
        )


class CarliniOpenAIInstructChessPipelineWithTimeControlAndElo(
    CarliniOpenAIInstructChessPipeline
):
    """Carlini and Paleka's OpenAI Instruct Chess Pipeline,
    except this one uses the actual time control and Elo in the prompt."""

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return CarliniPipelineConfig(
            input_mode=CarliniInputMode.TIME_CONTROL_AND_ELO
        ).input_config

    def make_prediction(
        self,
        pgns: list[str] | str,
        time_control: list[str] | str,
        white_elo: list[int] | int,
        black_elo: list[int] | int,
        num_tokens: int | None = None,
        verbose: bool = False,
        tqdm_leave: bool = False,
        return_histories: bool = False,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        return super().make_prediction(
            pgns,
            num_tokens,
            time_controls=time_control,
            white_elos=white_elo,
            black_elos=black_elo,
            verbose=verbose,
            tqdm_leave=tqdm_leave,
            return_histories=return_histories,
            num_workers=num_workers,
            parallel_threshold=parallel_threshold,
            **kwargs,
        )


# Note: the OpenAI client keeps getting throttled, so we just use the synchronous one because it's faster
class AsyncCarliniOpenAIInstructChessPipeline(CarliniOpenAIInstructChessPipeline):
    """Carlini and Paleka's OpenAI Instruct Chess Pipeline."""

    def __init__(
        self,
        model_id: str,
        num_lookahead_tokens: int | None = None,
        cache_path_template: str = "~/.cache/openai_instruct_chess/{model_id}/cache.json",
        warn_on_game_over: bool | None = None,
        pgn_header_override: str | None = None,
        input_config: PipelineInputConfig | None = None,
        pipeline_config: CarliniPipelineConfig | None = None,
    ):
        super().__init__(
            model_id=model_id,
            num_lookahead_tokens=num_lookahead_tokens,
            cache_path_template=cache_path_template,
            warn_on_game_over=warn_on_game_over,
            pgn_header_override=pgn_header_override,
            client_factory=openai.AsyncOpenAI,
            input_config=input_config,
            pipeline_config=pipeline_config,
        )

    def make_prediction(
        self,
        pgns: list[str] | str,
        num_tokens: int | None = None,
        verbose: bool = False,
        tqdm_leave: bool = False,
        return_histories: bool = False,
        batch_size: int = 16,
        persistent_workers: bool = False,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        prompts: list[str] | str | None = None,
        time_controls: list[str] | str | None = None,
        white_elos: list[int] | int | None = None,
        black_elos: list[int] | int | None = None,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        self.input_config.validate(
            pgns=pgns,
            prompts=prompts,
            time_controls=time_controls,
            white_elos=white_elos,
            black_elos=black_elos,
            **extract_domain_inputs(kwargs),
        )
        num_tokens = self._resolve_num_tokens(num_tokens)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        item_count = 1 if isinstance(pgns, str) else len(pgns)
        resolved_workers = resolve_num_workers(
            item_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )

        dataset = self._make_dataset(
            pgns,
            prompts=prompts,
            time_controls=time_controls,
            white_elos=white_elos,
            black_elos=black_elos,
            num_workers=resolved_workers,
            parallel_threshold=parallel_threshold,
        )
        raw_outputs = []
        for prompts, histories, first_fen_indices, seen_fen_indices in progress(
            iter_dataset_batches(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset.async_collate_fn,
                num_workers=resolved_workers,
                persistent_workers=persistent_workers,
            ),
            desc="Making predictions",
            total=(len(dataset) + batch_size - 1) // batch_size,
            leave=tqdm_leave,
        ):
            # Identify which prompts need to be queried
            indices_to_query = []
            prompts_to_query = []
            boards_to_query = []
            batch_outputs = {}
            for first_fen_index, prompt in zip(first_fen_indices, prompts):
                history = histories[first_fen_index]
                board = history.active_board
                if board.fen() in self.cache:
                    out = self.cache[board.fen()]
                    if verbose:
                        if board.ply() > 0:
                            logging.debug(
                                f"LLM played a move already in my cache. Returning {out}.",
                            )
                    batch_outputs[first_fen_index] = out
                else:
                    indices_to_query.append(first_fen_index)
                    prompts_to_query.append(prompt)
                    boards_to_query.append(board)

                    if verbose:
                        logging.debug(
                            f"Querying {self.model_id} with ... {prompt.split(']')[-1][-90:]}",
                        )

            responses = asyncio_run(
                self.forward(
                    prompts_to_query,
                    boards_to_query,
                    num_tokens,
                    verbose=verbose,
                )
            )
            if len(responses) != len(indices_to_query):
                raise ValueError(
                    "Async text-generation backend returned a different number "
                    "of responses than uncached inputs"
                )

            # Add the responses to the cache and batch outputs
            for index, next_moves, board in zip(
                indices_to_query, responses, boards_to_query
            ):
                if next_moves is None:
                    batch_outputs[index] = None
                    continue
                new_board = board.copy()
                for move in next_moves:
                    self.cache[new_board.fen()] = move
                    new_board.push_san(move)
                batch_outputs[index] = next_moves[0]

            # Fill in outputs for seen FENs
            for i in seen_fen_indices:
                history = histories[i]
                board = history.active_board
                out = self.cache.get(board.fen())
                if out is None:
                    warnings.warn(
                        f"Obtained invalid move when looking up seen FEN within the same batch. FEN={board.fen()}",
                        stacklevel=2,
                    )
                batch_outputs[i] = out

            # Collect outputs in the original order
            batch_next_moves = [batch_outputs[i] for i in range(len(histories))]
            if return_histories:
                raw_outputs.extend(zip(histories, batch_next_moves))
            else:
                boards = [history.active_board for history in histories]
                raw_outputs.extend(zip(boards, batch_next_moves))
        with self.cache_path.open("w") as f:
            json.dump(self.cache, f)

        if raw_outputs:
            predictions = adaptive_process_map(
                self.post_process_predictions,
                *zip(*raw_outputs),
                parallel_threshold=parallel_threshold,
                max_workers=resolved_workers,
                desc="Collecting predictions",
                leave=tqdm_leave,
                chunksize=64,
            )
        else:
            predictions = []
        return ChessMovePredictionDataset[ChessMovePrediction](predictions=predictions)

    async def send_message(self, prompt: str, num_tokens: int) -> str | list[str]:
        try:
            response = await self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                temperature=0.0,
                max_tokens=num_tokens,
            )
        except openai.BadRequestError as e:
            logging.error(f"Error from OpenAI API: {e}")
            logging.error(f"Prompt that caused the error: {prompt}")
            raise e

        messages = [choice.text for choice in response.choices]

        if len(messages) == 1:
            return messages[0]
        else:
            return [message for message in messages]

    async def predict_move(
        self, prompt: str, board: chess.Board, num_tokens: int, verbose: bool = False
    ) -> list[str] | None:
        next_text = await self.send_message(prompt, num_tokens)
        if not isinstance(next_text, str):
            raise ValueError(
                f"Expected a single string response from the model. {next_text=}"
            )

        if next_text[:2] == "-O":
            next_text = await self.send_message(prompt + " ", num_tokens)
        if not isinstance(next_text, str):
            raise ValueError(
                f"Expected a single string response from the model. {next_text=}"
            )

        if verbose:
            logging.debug(f"Received reply of '{next_text}'")

        next_moves = self.try_moves(board, next_text)

        if len(next_moves) == 0:
            logging.debug("Tried to make an invalid move.")
            return None

        if verbose:
            logging.debug(f"Received reply and making move {next_moves[0]}.")

        return next_moves

    async def forward(
        self,
        prompts: list[str],
        boards: list[chess.Board],
        num_tokens: int,
        verbose: bool = False,
    ) -> list[list[str] | None]:
        return await asyncio.gather(
            *[
                self.predict_move(prompt, board, num_tokens, verbose)
                for prompt, board in zip(prompts, boards)
            ]
        )


MAX_RETRIES = 5

CarliniPipeline = CarliniOpenAIInstructChessPipeline
__all__ = [
    "AsyncCarliniOpenAIInstructChessPipeline",
    "CarliniOpenAIInstructChessDataset",
    "CarliniOpenAIInstructChessDatasetWithPrompt",
    "CarliniOpenAIInstructChessDatasetWithTimeControlAndElo",
    "CarliniOpenAIInstructChessPipeline",
    "CarliniPipeline",
    "CarliniPipelineConfig",
    "CarliniInputMode",
    "CarliniOpenAIInstructChessPipelineWithPrompt",
    "CarliniOpenAIInstructChessPipelineWithTimeControlAndElo",
    "DEFAULT_CARLINI_PGN_HEADER_BLOG",
    "DEFAULT_CARLINI_PGN_HEADER_CODE",
    "DEFAULT_CARLINI_NUM_LOOKAHEAD_TOKENS",
    "DEFAULT_PGN_HEADER_WITH_TIME_CONTROL_AND_ELO",
    "MAX_RETRIES",
]
