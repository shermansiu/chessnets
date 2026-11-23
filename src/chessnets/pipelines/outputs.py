"""Shared prediction result types and prediction collections."""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import typing as tp
import warnings
import types

import chess

from chessnets.models.allie import data as allie_data
from chessnets.utils import board_history
from chessnets._optional import optional_module
from .pipeline_utils import progress

IPython = types.SimpleNamespace(display=optional_module("IPython.display", "notebook"))


@dataclasses.dataclass
class ChessMove:
    move: str
    probability: float


@dataclasses.dataclass
class ChessResponse:
    moves: list[ChessMove]

    def to_best_moves_and_preds_list(
        self,
        board: chess.Board,
        rescale_excess: bool = False,
        fill_missing: bool = False,
    ):
        preds_list = []
        for move in self.moves:
            move_san = move.move.rsplit(" ", 1)[-1].rsplit(".", 1)[-1]
            try:
                move_uci = board.parse_san(move_san)
                preds_list.append((move_uci, move.probability))
            except chess.InvalidMoveError:
                # Ignore invalid moves
                logging.error(
                    f"Invalid move from model: {move_san} in position {board.fen()}"
                )
                continue
            except chess.IllegalMoveError:
                # Ignore illegal moves
                logging.error(
                    f"Illegal move from model: {move_san} in position {board.fen()}"
                )
                continue
            except chess.AmbiguousMoveError:
                # Ignore ambiguous moves
                logging.error(
                    f"Ambiguous move from model: {move_san} in position {board.fen()}"
                )
                continue
        if rescale_excess:
            total_prob = sum(prob for _, prob in preds_list)
            if total_prob > 1.0:
                preds_list = [(move, prob / total_prob) for move, prob in preds_list]
        if fill_missing:
            legal_moves = set(board.legal_moves)
            predicted_moves = set(move for move, _ in preds_list)
            missing_moves = legal_moves - predicted_moves
            total_prob = sum(prob for _, prob in preds_list)
            if missing_moves and total_prob < 1.0:
                missing_prob = 1.0 - total_prob
                prob_per_missing_move = missing_prob / len(missing_moves)
                for move in missing_moves:
                    preds_list.append((move, prob_per_missing_move))
        preds_list.sort(key=lambda x: x[1], reverse=True)
        best_move_uci = str(preds_list[0][0])
        best_move_san = board.san(preds_list[0][0])
        return best_move_san, best_move_uci, dict(preds_list)


@dataclasses.dataclass
class ChessMovePrediction:
    board_history: board_history.BoardHistory | None
    best_move_san: str
    best_move_uci: str
    preds_list: dict[chess.Move, float]
    win_probability: float | None = None
    result_probs: dict[str, float] | None = None
    result: str | None = None
    moves_left: float | None = None

    def notebook_display(self):
        if self.board_history is not None:
            print(f"Active board after {len(self.board_history.moves)} moves:")
            board = self.board_history.active_board
            IPython.display.display(board)
        print(f"Best move: {self.best_move_san}")
        print()

        print("All legal moves with probabilities:")
        if self.board_history is not None:
            before_move_board = self.board_history.positions[-2].board
            for move, prob in self.preds_list.items():
                print(f"{before_move_board.san(move)}: {prob:.4f}")
            print()
        else:
            for move, prob in self.preds_list.items():
                print(f"{move}: {prob:.4f}")
            print()

        if self.win_probability is not None:
            print(f"Win probability: {self.win_probability:.4f}")

        if self.moves_left is not None:
            print(f"Estimated moves left until game end: {self.moves_left:.1f}")

        if self.board_history is not None:
            print("PGN:")
            print(self.board_history.to_pgn())


@dataclasses.dataclass
class AuxiliaryTargetChessMovePrediction(ChessMovePrediction):
    outcome_type_probs: dict[str, float] | None = None
    outcome_type: str | None = None
    move_delay: float | None = None


@dataclasses.dataclass
class LLMChessMovePrediction(ChessMovePrediction):
    raw_response: str | None = None


@dataclasses.dataclass
class Maia2ChessMovePrediction:
    leela_board: board_history.LeelaBoard
    best_move_san: str
    best_move_uci: str
    preds_list: dict[chess.Move, float]
    win_probability: float

    def notebook_display(self):
        print("Active board:")

        board = self.leela_board.board.copy()
        board.push_san(self.best_move_san)
        IPython.display.display(board)
        print(f"Best move: {self.best_move_san}")
        print(f"Win probability: {self.win_probability:.4f}")
        print()

        print("All legal moves with probabilities:")
        before_move_board = self.leela_board.board
        for move, prob in self.preds_list.items():
            print(f"{before_move_board.san(move)}: {prob:.4f}")
        print()

        print("FEN:")
        print(self.leela_board.board.fen())


@dataclasses.dataclass
class AllieMovePrediction:
    game: allie_data.Game
    best_move_san: str
    best_move_uci: str
    preds_list_legal: dict[chess.Move, float]
    preds_list: dict[chess.Move, float]
    resigned: bool
    win_probability: float
    think_time: float | None
    is_top_move_legal: bool
    legal_move_probability: float

    def notebook_display(self):
        print(f"Active board after {len(self.game.moves)} moves:")

        board = self.game.board
        IPython.display.display(board)
        print(f"Best move: {self.best_move_san}")
        print()

        print("All legal moves with probabilities:")
        before_move_board = board.copy()
        for move, prob in self.preds_list_legal.items():
            print(f"{before_move_board.san(move)}: {prob:.4f}")
        print()

        print("All moves (including illegal) with probabilities:")
        for move, prob in self.preds_list.items():
            print(f"{before_move_board.san(move)}: {prob:.4f}")
        print()

        print("PGN:")
        game = chess.pgn.Game()
        node = game
        for move in self.game.moves:
            node = node.add_variation(chess.Move.from_uci(move))
        exporter = chess.pgn.StringExporter(
            headers=False, variations=False, comments=False
        )
        pgn_string = game.accept(exporter).removesuffix(" *")
        print(pgn_string)


MovePrediction = ChessMovePrediction | Maia2ChessMovePrediction | AllieMovePrediction
T = tp.TypeVar("T")


@dataclasses.dataclass
class ChessMovePredictionDataset(tp.Generic[T]):
    predictions: list[T]

    def notebook_display(self):
        for i, prediction in enumerate(self.predictions):
            print(f"Prediction {i + 1}:")
            prediction.notebook_display()
            print("=" * 40 + "\n")

    @functools.cached_property
    def generic_type_arg(self):
        # Accessing the original type annotation from the instance's class
        # __orig_bases__ is a tuple of the original base classes (e.g., Generic[int])
        # get_args extracts the type arguments from the generic base
        if not hasattr(self, "__orig_class__"):
            return None
        return tp.get_args(self.__orig_class__)[0]

    @functools.cached_property
    def best_move_sans(self) -> list[str]:
        return [
            prediction.best_move_san
            for prediction in progress(
                self.predictions, desc="Extracting best move SANs", leave=False
            )
        ]

    @functools.cached_property
    def best_move_ucis(self) -> list[str]:
        return [
            prediction.best_move_uci
            for prediction in progress(
                self.predictions, desc="Extracting best move UCIs", leave=False
            )
        ]

    @functools.cached_property
    def best_move_probabilities(self) -> list[float]:
        return [
            prediction.preds_list[chess.Move.from_uci(prediction.best_move_uci)]
            for prediction in progress(
                self.predictions, desc="Extracting best move probabilities", leave=False
            )
        ]

    @functools.cached_property
    def win_probabilities(self) -> list[float | None] | None:
        if issubclass(
            self.generic_type_arg,
            (ChessMovePrediction, Maia2ChessMovePrediction, AllieMovePrediction),
        ):
            return [
                prediction.win_probability
                for prediction in progress(
                    self.predictions, desc="Extracting win probabilities", leave=False
                )
            ]
        if self.generic_type_arg is not None:
            return None
        return [
            (
                prediction.win_probability
                if hasattr(prediction, "win_probability")
                else None
            )
            for prediction in progress(
                self.predictions, desc="Extracting win probabilities", leave=False
            )
        ]

    @functools.cached_property
    def results(self) -> list[str | None] | None:
        if issubclass(self.generic_type_arg, ChessMovePrediction):
            return [
                prediction.result
                for prediction in progress(
                    self.predictions, desc="Extracting results", leave=False
                )
            ]
        if self.generic_type_arg is not None:
            return None
        return [
            prediction.result if hasattr(prediction, "result") else None
            for prediction in progress(
                self.predictions, desc="Extracting results", leave=False
            )
        ]

    @functools.cached_property
    def pred_result_probabilities(self) -> list[float | None] | None:
        if issubclass(self.generic_type_arg, ChessMovePrediction):
            return [
                prediction.result_probs[prediction.result]
                for prediction in progress(
                    self.predictions,
                    desc="Extracting predicted result probabilities",
                    leave=False,
                )
            ]
        if self.generic_type_arg is not None:
            return None
        return [
            (
                prediction.result_probs[prediction.result]
                if hasattr(prediction, "result_probs") and hasattr(prediction, "result")
                else None
            )
            for prediction in progress(
                self.predictions,
                desc="Extracting predicted result probabilities",
                leave=False,
            )
        ]

    @functools.cached_property
    def moves_left(self) -> list[float | None] | None:
        if issubclass(self.generic_type_arg, ChessMovePrediction):
            return [
                prediction.moves_left
                for prediction in progress(
                    self.predictions,
                    desc="Extracting moves left estimates",
                    leave=False,
                )
            ]
        if self.generic_type_arg is not None:
            return None
        return [
            prediction.moves_left if hasattr(prediction, "moves_left") else None
            for prediction in progress(
                self.predictions, desc="Extracting moves left estimates", leave=False
            )
        ]

    @functools.cached_property
    def outcomes(self) -> list[str | None] | None:
        if issubclass(self.generic_type_arg, AuxiliaryTargetChessMovePrediction):
            return [
                prediction.outcome_type
                for prediction in progress(
                    self.predictions, desc="Extracting outcome types", leave=False
                )
            ]
        if self.generic_type_arg is not None:
            return None
        return [
            prediction.outcome_type if hasattr(prediction, "outcome_type") else None
            for prediction in progress(
                self.predictions, desc="Extracting outcome types", leave=False
            )
        ]

    @functools.cached_property
    def pred_outcome_probabilities(self) -> list[float | None] | None:
        if issubclass(self.generic_type_arg, AuxiliaryTargetChessMovePrediction):
            return [
                prediction.outcome_type_probs[prediction.outcome_type]
                for prediction in progress(
                    self.predictions,
                    desc="Extracting predicted outcome probabilities",
                    leave=False,
                )
            ]
        if self.generic_type_arg is not None:
            return None
        return [
            (
                prediction.outcome_type_probs[prediction.outcome_type]
                if hasattr(prediction, "outcome_type_probs")
                and hasattr(prediction, "outcome_type")
                else None
            )
            for prediction in progress(
                self.predictions,
                desc="Extracting predicted outcome probabilities",
                leave=False,
            )
        ]

    @functools.cached_property
    def has_resigned(self) -> list[bool | None] | None:
        if issubclass(self.generic_type_arg, AuxiliaryTargetChessMovePrediction):
            return [
                prediction.outcome_type == "Resigned"
                for prediction in progress(
                    self.predictions,
                    desc="Extracting resignation flags",
                    leave=False,
                )
            ]
        if issubclass(self.generic_type_arg, AllieMovePrediction):
            return [
                prediction.resigned
                for prediction in progress(
                    self.predictions,
                    desc="Extracting resignation flags",
                    leave=False,
                )
            ]
        return None

    @functools.cached_property
    def pred_has_resigned_probabilities(self) -> list[float | None] | None:
        if issubclass(self.generic_type_arg, AuxiliaryTargetChessMovePrediction):
            return [
                (
                    prediction.outcome_type_probs.get("Resigned", 0.0)
                    if hasattr(prediction, "outcome_type_probs")
                    else None
                )
                for prediction in progress(
                    self.predictions,
                    desc="Extracting predicted resignation probabilities",
                    leave=False,
                )
            ]
        if issubclass(self.generic_type_arg, AllieMovePrediction):
            warnings.warn(
                "Predicted resignation probabilities are not implemented for AllieMovePrediction. Returning None."
            )
        return None

    @functools.cached_property
    def move_delays(self) -> list[float | None] | None:
        if issubclass(self.generic_type_arg, AuxiliaryTargetChessMovePrediction):
            return [
                prediction.move_delay
                for prediction in progress(
                    self.predictions, desc="Extracting move delays", leave=False
                )
            ]
        if issubclass(self.generic_type_arg, AllieMovePrediction):
            return [
                prediction.think_time
                for prediction in progress(
                    self.predictions, desc="Extracting move delays", leave=False
                )
            ]
        if self.generic_type_arg is not None:
            return None
        return [
            prediction.move_delay if hasattr(prediction, "move_delay") else None
            for prediction in progress(
                self.predictions, desc="Extracting move delays", leave=False
            )
        ]

    @functools.cached_property
    def raw_responses(self) -> list[str] | None:
        if self.generic_type_arg is not None and not issubclass(
            self.generic_type_arg, LLMChessMovePrediction
        ):
            return None
        return [
            prediction.raw_response
            for prediction in progress(
                self.predictions, desc="Extracting raw responses", leave=False
            )
        ]

    def target_move_uci_probabilities(self, move_ucis: list[str]) -> list[float]:
        if len(move_ucis) != len(self.predictions):
            raise ValueError(
                f"Length of move_sans ({len(move_ucis)}) does not match length of predictions ({len(self.predictions)})"
            )
        return [
            prediction.preds_list.get(chess.Move.from_uci(move_uci), 0.0)
            for prediction, move_uci in progress(
                zip(self.predictions, move_ucis),
                total=len(self.predictions),
                desc="Extracting best move probabilities",
                leave=False,
            )
        ]

    def target_result_probabilities(self, results: list[str]) -> list[float | None]:
        if len(results) != len(self.predictions):
            raise ValueError(
                f"Length of results ({len(results)}) does not match length of predictions ({len(self.predictions)})"
            )
        return [
            (
                prediction.result_probs.get(result, 0.0)
                if hasattr(prediction, "result_probs")
                else None
            )
            for prediction, result in progress(
                zip(self.predictions, results),
                total=len(self.predictions),
                desc="Extracting target result probabilities",
                leave=False,
            )
        ]

    def target_outcome_probabilities(self, outcomes: list[str]) -> list[float | None]:
        if len(outcomes) != len(self.predictions):
            raise ValueError(
                f"Length of outcomes ({len(outcomes)}) does not match length of predictions ({len(self.predictions)})"
            )
        return [
            (
                prediction.outcome_type_probs.get(outcome, 0.0)
                if hasattr(prediction, "outcome_type_probs")
                else None
            )
            for prediction, outcome in progress(
                zip(self.predictions, outcomes),
                total=len(self.predictions),
                desc="Extracting target outcome probabilities",
                leave=False,
            )
        ]

    def target_has_resigned_probabilities(
        self, has_resigned_flags: list[bool]
    ) -> list[float | None]:
        if len(has_resigned_flags) != len(self.predictions):
            raise ValueError(
                f"Length of has_resigned_flags ({len(has_resigned_flags)}) does not match length of predictions ({len(self.predictions)})"
            )
        return [
            (
                (
                    prediction.outcome_type_probs.get("Resigned", 0.0)
                    if flag
                    else 1 - prediction.outcome_type_probs.get("Resigned", 0.0)
                )
                if hasattr(prediction, "outcome_type_probs")
                and flag is not None
                and not math.isnan(flag)
                else None
            )
            for prediction, flag in progress(
                zip(self.predictions, has_resigned_flags),
                total=len(self.predictions),
                desc="Extracting target resignation probabilities",
                leave=False,
            )
        ]


__all__ = [
    "AllieMovePrediction",
    "AuxiliaryTargetChessMovePrediction",
    "ChessMove",
    "ChessMovePrediction",
    "ChessMovePredictionDataset",
    "ChessResponse",
    "LLMChessMovePrediction",
    "Maia2ChessMovePrediction",
    "MovePrediction",
]
