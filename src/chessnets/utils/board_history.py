import copy
import dataclasses
import io
import typing as tp
import attrs
import chess
import chess.pgn
import numpy as np
from chessnets.utils import decode_policy
from chessnets.models.maia2 import utils as maia2_utils


def bitboards_to_array_display(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)


def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    """Returns the bitboards as a numpy array.

    Right-to-left, then bottom-to-top. h1 is 0, h2 is 8, and a8 is 63.
    """
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(8, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 64)


def board_to_array(board: chess.Board):
    black, white = board.occupied_co

    bitboards = np.array(
        [
            white & board.pawns,
            white & board.knights,
            white & board.bishops,
            white & board.rooks,
            white & board.queens,
            white & board.kings,
            black & board.pawns,
            black & board.knights,
            black & board.bishops,
            black & board.rooks,
            black & board.queens,
            black & board.kings,
        ],
        dtype=np.uint64,
    )

    return bitboards_to_array(bitboards)


def board_fen(board: chess.Board):
    """Return the board FEN, without the halfmove clock and fullmove number."""
    return board.fen().rsplit(" ", 2)[0]


PositionKey = tuple[int, int, int, int, int, int, int, int, bool, int, int | None]


def position_key(board: chess.Board) -> PositionKey:
    """Return a hashable key identifying a position for repetition detection.

    Equivalent to ``board_fen(board)`` (pieces, side to move, castling rights,
    and an en passant square only when a legal capture exists), but built from
    bitboards, which is several times faster than formatting a FEN string.
    """
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.turn,
        board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


def bool_to_plane(val: bool, dtype=np.float32) -> np.ndarray:
    """Convert a boolean value to a Lc0 plane."""
    return np.full((1, 64), val, dtype=dtype)


@attrs.define
class LeelaBoard:
    board: chess.Board
    repetitions: int = -1
    plies_since_prev_rep: int = -1
    turn: int = attrs.field(init=False)
    ply: int = attrs.field(init=False)

    @classmethod
    def from_fen(cls, fen: str) -> tp.Self:
        board = chess.Board(fen)
        return cls(board)

    @property
    def is_white_turn(self) -> bool:
        return self.turn == chess.WHITE

    @property
    def is_black_turn(self) -> bool:
        return self.turn == chess.BLACK

    @property
    def parity(self) -> int:
        return 0 if self.is_white_turn else 1

    def __attrs_post_init__(self):
        self.ply = self.board.ply()
        # ``Board.turn`` is already part of the position state.  Deriving the
        # side to move from the ply is both unnecessary and incorrect for a
        # FEN whose fullmove number is not consistent with its turn field.
        self.turn = self.board.turn

    def asarray(self, mirror: bool = False) -> tuple[np.ndarray, np.ndarray]:
        board = self.board.mirror() if mirror else self.board
        return board_to_array(board), bool_to_plane(
            self.repetitions > 0
        )  # Assume automatic draw upon threefold repetition

    def to_aux_plane(self) -> np.ndarray:
        vals = np.asarray(
            [
                self.board.has_queenside_castling_rights(chess.WHITE),
                self.board.has_kingside_castling_rights(chess.WHITE),
                self.board.has_queenside_castling_rights(chess.BLACK),
                self.board.has_kingside_castling_rights(chess.BLACK),
                self.is_black_turn,
                self.board.halfmove_clock
                / 99,  # Normalized during training, not Lc0 inference for non-hectoplies format
                False,  # No longer used, all zeros
                True,  # All ones, to help Lc0 find board edges
            ],
            dtype=np.float32,
        ).reshape(-1, 1)
        # Broadcasting + unnecessary multiplication is faster than np.repeat
        return np.ones((8, 64), dtype=np.float32) * vals

    def to_maia2_array(self) -> np.ndarray:
        board = self.board.mirror() if self.is_black_turn else self.board
        board_array = board_to_array(board)

        aux_array_vals = np.asarray(
            [
                True,  # The official implementation checks whether it's white's turn, but the board is mirrored when it's black's turn, so this is always true
                board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK),
            ],
            dtype=np.float32,
        ).reshape(-1, 1)
        # Broadcasting + unnecessary multiplication is faster than np.repeat
        aux_array = np.ones((5, 64), dtype=np.float32) * aux_array_vals

        ep_square_plane = np.zeros((1, 64), dtype=np.float32)
        if board.ep_square is not None:
            ep_square_plane[0, board.ep_square] = 1.0
        return np.vstack([board_array, aux_array, ep_square_plane]).reshape(-1, 8, 8)

    def to_maia2_legal_moves_mask(self) -> np.ndarray:
        board = self.board.mirror() if self.is_black_turn else self.board
        legal_moves_mask = np.zeros(len(maia2_utils.MOVES), dtype=bool)
        legal_moves_idx = np.asarray(
            [maia2_utils.MOVE_TO_INDEX[move.uci()] for move in board.legal_moves]
        )
        legal_moves_mask[legal_moves_idx] = 1
        return legal_moves_mask


MOVE_HISTORY = 8
PLANES_PER_BOARD = 13
AUX_PLANE_BASE = PLANES_PER_BOARD * MOVE_HISTORY


@dataclasses.dataclass
class TimeInfo:
    time_control: str
    active_player_last_move_time_spent: float
    opponent_last_move_time_spent: float
    active_player_current_move_time_spent: float
    active_player_time_remaining: float
    opponent_time_remaining: float


@attrs.define
class BoardHistory:
    active_board: chess.Board = attrs.field(factory=chess.Board)
    positions: list[LeelaBoard] = attrs.field(init=False)
    moves: list[chess.Move] = attrs.field(factory=list)
    # Values are references to snapshots already owned by ``positions``.  No
    # additional chess.Board (or move stack) is created for repetition data.
    last_seen_table: tuple[
        dict[PositionKey, LeelaBoard], dict[PositionKey, LeelaBoard]
    ] = attrs.field(factory=lambda: (dict(), dict()))
    result: str | None = attrs.field(default=None)
    time_control: str | None = attrs.field(default=None)
    white_clock_times: list[float | None] | None = attrs.field(default=None)
    black_clock_times: list[float | None] | None = attrs.field(default=None)

    def _snapshot(self) -> LeelaBoard:
        """Copy the current state without copying python-chess move history."""
        return LeelaBoard(self.active_board.copy(stack=False))

    def __attrs_post_init__(self):
        init_board = self._snapshot()
        init_board.repetitions = 0
        self.positions: list[LeelaBoard] = [init_board]

        # The initial position is an occurrence too.  Seeding the table makes
        # a return to the initial position count correctly, including when the
        # history starts from a non-standard FEN/fullmove number.
        self.last_seen_table[init_board.parity][position_key(init_board.board)] = (
            init_board
        )

    def _compute_last_move_repetitions(self) -> tuple[int, int]:
        last = self.positions[-1]
        if last.board.halfmove_clock < 4:
            return 0, -1

        last_seen = self.last_seen_table[last.parity].get(position_key(last.board))

        if last_seen is None:
            return 0, -1

        return last_seen.repetitions + 1, last.ply - last_seen.ply

    def __len__(self):
        """Returns the number of moves in the history."""
        return len(self.moves)

    def append(self, move: chess.Move):
        self.active_board.push(move)
        self.moves.append(move)

        last = self._snapshot()
        self.positions.append(last)

        repetitions, plies_since_prev_rep = self._compute_last_move_repetitions()
        last.repetitions = repetitions
        last.plies_since_prev_rep = plies_since_prev_rep

        self.last_seen_table[last.parity][position_key(last.board)] = last

    def fork(self) -> tp.Self:
        """Return an independent history sharing immutable position snapshots.

        ``active_board`` is copied with its move stack because callers may
        continue to use ``push``/``pop`` on the fork.  Historical snapshots
        are never mutated by ``BoardHistory``, so the positions and repetition
        values can be shared while their containers are copied.
        """
        history = copy.copy(self)
        history.active_board = self.active_board.copy(stack=True)
        history.moves = self.moves.copy()
        history.positions = self.positions.copy()
        history.last_seen_table = (
            self.last_seen_table[0].copy(),
            self.last_seen_table[1].copy(),
        )
        if self.white_clock_times is not None:
            history.white_clock_times = self.white_clock_times.copy()
        if self.black_clock_times is not None:
            history.black_clock_times = self.black_clock_times.copy()
        return history

    def copy_with_move(self, move: chess.Move) -> tp.Self:
        """Return a fork with ``move`` appended, without deep-copying history."""
        history = self.fork()
        history.append(move)
        return history

    @classmethod
    def from_game(cls, game: chess.pgn.Game, should_load_clock_info: bool = False):
        # ``game.board()`` honours the SetUp/FEN (and Variant) headers, so games
        # that start from a custom position are replayed from that position.
        start = game.board()
        history = cls(active_board=start)

        if should_load_clock_info:
            history.white_clock_times = []
            history.black_clock_times = []

        # Get the mainline moves
        node = game
        is_black = start.turn == chess.BLACK
        while not node.is_end():
            next_node = node.variation(0)  # Move to the next node in the mainline

            history.append(next_node.move)

            # Get the clock time for the move that led to this node
            if should_load_clock_info:
                assert (
                    history.white_clock_times is not None
                    and history.black_clock_times is not None
                )
                time_remaining = next_node.clock()
                if is_black:
                    history.black_clock_times.append(time_remaining)
                else:
                    history.white_clock_times.append(time_remaining)

            node = next_node
            is_black = not is_black

        history.result = game.headers["Result"]
        history.time_control = game.headers.get("TimeControl")

        return history

    @classmethod
    def from_pgn(cls, pgn: str, should_load_clock_info: bool = False):
        if pgn:
            game = chess.pgn.read_game(io.StringIO(pgn))
        else:
            game = chess.pgn.Game()
        return cls.from_game(game, should_load_clock_info=should_load_clock_info)

    @property
    def move_indices(self) -> list[int]:
        return [
            decode_policy.get_move_index(move, position.is_black_turn)
            for move, position in zip(self.moves, self.positions)
        ]

    def move_index(self, turn_index: int = -1) -> int:
        if turn_index < 0:
            turn_index += len(self.moves)
        position = self.positions[turn_index]  # Length is len(moves) + 1
        return decode_policy.get_move_index(
            self.moves[turn_index], should_mirror=position.is_black_turn
        )

    def legal_move_indices(self, turn_position_index: int = -1) -> list[int]:
        """Returns the legal move indices from the perspective of the current player."""
        position = self.positions[turn_position_index]
        return sorted(
            decode_policy.get_move_index(move, should_mirror=position.is_black_turn)
            for move in position.board.legal_moves
        )

    def move_tensor_target(self, turn_index: int = -1) -> np.ndarray:
        """Returns the move tensor target.

        Args:
            index: The move index

        Returns:
            An int ndarray of shape (move_dim,), with a 1 for the target move,
                a 0 for all other legal moves, and -1 for all illegal moves.
        """
        if turn_index < 0:
            turn_index += len(self.moves)
        move_index = self.move_index(turn_index)
        legal_move_indices = self.legal_move_indices(
            turn_index
        )  # Careful: using a negative index will cause an off-by-one error
        if move_index not in legal_move_indices:
            raise ValueError(
                f"Move {move_index} is not listed amongst the legal move indices {legal_move_indices}"
            )
        target = np.full(
            shape=len(decode_policy.MOVES), fill_value=-1
        )  # -1 = illegal move
        target[legal_move_indices] = 0
        target[move_index] = 1
        return target

    def asarray(self, last_position_index: int = -1) -> np.ndarray:
        """Converts a chess board to a Lc0 input board.

        Adapted from https://github.com/LeelaChessZero/lc0/blob/master/src/neural/encoder.cc
        and https://github.com/dkappe/badgyal/blob/master/badgyal/board2planes.py
        """
        if not self.positions:
            raise ValueError("History is empty. Cannot create lc0 input.")

        if last_position_index < 0:
            last_position_index += len(self.positions)
        if not 0 <= last_position_index < len(self.positions):
            raise IndexError("last_position_index is outside the board history")

        # Only eight positions can contribute to an Lc0 input.  Avoid copying
        # the entire history when encoding an old position in a long game.
        start = max(0, last_position_index - MOVE_HISTORY + 1)
        positions = self.positions[start : last_position_index + 1]

        planes = []
        we_are_black = positions[-1].is_black_turn
        for lboard in reversed(positions):  # 8 most recent, in reverse order
            planes.extend(lboard.asarray(we_are_black))
        num_games = len(planes) // 2
        if num_games < MOVE_HISTORY:
            planes.append(
                np.zeros(
                    (PLANES_PER_BOARD * (MOVE_HISTORY - num_games), 64),
                    dtype=np.float32,
                )
            )

        planes.append(positions[-1].to_aux_plane())

        return np.vstack(planes)

    def to_maia2_array(self, position_index: int = -1) -> np.ndarray:
        """Converts a chess board to a Maia2 input board."""
        return self.positions[position_index].to_maia2_array()

    def to_game(self, until_turn_index: int = -1) -> chess.pgn.Game:
        """Convert the board history to a Python-chess game."""
        game = chess.pgn.Game()
        game.setup(self.positions[0].board)  # Adds SetUp/FEN for custom starts

        if not self.moves or until_turn_index == 0:
            return game

        node = game.add_variation(self.moves[0])
        if until_turn_index < 0:
            until_turn_index = len(self.positions) + until_turn_index
        for move in self.moves[1:until_turn_index]:
            node = node.add_variation(move)

        return game

    def to_pgn(self, until_turn_index: int = -1) -> str:
        """Convert the board history to a PGN string."""
        game = self.to_game(until_turn_index=until_turn_index)
        if game.end().board().is_game_over(claim_draw=True) and self.result:
            game.headers["Result"] = self.result
        exporter = chess.pgn.StringExporter(
            headers=True, variations=False, comments=False
        )
        return game.accept(exporter).removesuffix(" *")

    def to_pgn_moves(self, until_turn_index: int = -1) -> str:
        """Convert the board history to a PGN moves string."""
        game = self.to_game(until_turn_index=until_turn_index)
        exporter = chess.pgn.StringExporter(
            headers=False,
            variations=False,
            comments=False,
            columns=None,
        )
        return game.accept(exporter).removesuffix(" *")

    @property
    def time_increment(self) -> int | None:
        if self.time_control is None:
            return None
        if "+" in self.time_control:
            return int(self.time_control.split("+")[1])
        return 0

    def get_time_info(self, turn_index: int):
        if turn_index < 0:
            turn_index += len(self.moves)
        white_clock_times = self.white_clock_times
        black_clock_times = self.black_clock_times
        if white_clock_times is None or black_clock_times is None:
            return None
        if self.positions[0].is_black_turn:
            # The indexing below assumes White moves first. For a game that
            # starts with Black to move, treat it as if White had already made
            # one move with an unknown clock reading.
            turn_index += 1
            white_clock_times = [None, *white_clock_times]
        a, b = divmod(turn_index, 2)
        white_index = a + b
        black_index = a
        is_black = b == 1

        increment = self.time_increment or 0

        if white_index >= len(white_clock_times) or black_index >= len(
            black_clock_times
        ):
            return None

        white_time_left = white_clock_times[white_index]
        black_time_left = black_clock_times[black_index]
        if (
            white_index > 0
            and white_clock_times[white_index - 1] is not None
            and white_time_left is not None
        ):
            white_last_turn_time_spent = (
                white_clock_times[white_index - 1] - white_time_left + increment
            )
        else:
            white_last_turn_time_spent = None
        if (
            black_index > 0
            and black_clock_times[black_index - 1] is not None
            and black_time_left is not None
        ):
            black_last_turn_time_spent = (
                black_clock_times[black_index - 1] - black_time_left + increment
            )
        else:
            black_last_turn_time_spent = None

        if is_black:
            if (
                black_index < len(black_clock_times) - 1
                and black_clock_times[black_index + 1] is not None
                and black_time_left is not None
            ):
                active_this_turn_time_spent = (
                    black_time_left - black_clock_times[black_index + 1] + increment
                )
            else:
                active_this_turn_time_spent = None

            return TimeInfo(
                time_control=self.time_control,
                active_player_last_move_time_spent=black_last_turn_time_spent,
                opponent_last_move_time_spent=white_last_turn_time_spent,
                active_player_current_move_time_spent=active_this_turn_time_spent,
                active_player_time_remaining=black_time_left,
                opponent_time_remaining=white_time_left,
            )
        else:
            if (
                white_index < len(white_clock_times) - 1
                and white_clock_times[white_index + 1] is not None
                and white_time_left is not None
            ):
                active_this_turn_time_spent = (
                    white_time_left - white_clock_times[white_index + 1] + increment
                )
            else:
                active_this_turn_time_spent = None

            return TimeInfo(
                time_control=self.time_control,
                active_player_last_move_time_spent=white_last_turn_time_spent,
                opponent_last_move_time_spent=black_last_turn_time_spent,
                active_player_current_move_time_spent=active_this_turn_time_spent,
                active_player_time_remaining=white_time_left,
                opponent_time_remaining=black_time_left,
            )
