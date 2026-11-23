"""Canonical live-game state and model-independent move results."""

from __future__ import annotations

import dataclasses
import typing as tp

import chess
import chess.pgn

from .matching import ExactTimeControl, time_control_from_clock


TERMINAL_STATUSES = frozenset(
    {
        "aborted",
        "outoftime",
        "resign",
        "mate",
        "stalemate",
        "draw",
        "timeout",
        "cheat",
        "noStart",
        "unknownFinish",
        "outoftime",
        "variantEnd",
        "variantLoss",
        "variantWin",
    }
)


@dataclasses.dataclass(frozen=True)
class PlayerInfo:
    id: str | None
    name: str
    rating: int | None = None


@dataclasses.dataclass
class LiveGameState:
    """Reducer state for one standard Lichess game."""

    game_id: str
    bot_username: str
    white: PlayerInfo
    black: PlayerInfo
    moves: list[str]
    initial_fen: str = "startpos"
    variant: str = "standard"
    speed: str | None = None
    rated: bool = False
    initial_clock_ms: int | None = None
    increment_ms: int | None = None
    wtime_ms: int | None = None
    btime_ms: int | None = None
    winc_ms: int | None = None
    binc_ms: int | None = None
    status: str = "started"
    white_clock_history: list[float | None] = dataclasses.field(default_factory=list)
    black_clock_history: list[float | None] = dataclasses.field(default_factory=list)
    move_seconds: list[int] = dataclasses.field(default_factory=list)

    @classmethod
    def from_game_full(
        cls, event: dict[str, tp.Any], bot_username: str
    ) -> "LiveGameState":
        if event.get("variant", {}).get("key", "standard") != "standard":
            raise ValueError("Only the standard chess variant is currently supported")
        clock = event.get("clock") or {}
        white = _player_info(event.get("white") or {})
        black = _player_info(event.get("black") or {})
        state = cls(
            game_id=str(event["id"]),
            bot_username=bot_username,
            white=white,
            black=black,
            moves=[],
            initial_fen=event.get("initialFen", "startpos"),
            variant=event.get("variant", {}).get("key", "standard"),
            speed=event.get("speed"),
            rated=bool(event.get("rated", False)),
            initial_clock_ms=clock.get("initial"),
            increment_ms=clock.get("increment"),
        )
        state.apply_game_state(event.get("state") or {})
        return state

    @property
    def bot_color(self) -> chess.Color:
        username = self.bot_username.casefold()
        if (self.white.id or self.white.name).casefold() == username:
            return chess.WHITE
        if (self.black.id or self.black.name).casefold() == username:
            return chess.BLACK
        raise ValueError(
            f"Bot account {self.bot_username!r} is not a player in game {self.game_id}"
        )

    @property
    def opponent(self) -> PlayerInfo:
        return self.black if self.bot_color == chess.WHITE else self.white

    @property
    def bot_player(self) -> PlayerInfo:
        return self.white if self.bot_color == chess.WHITE else self.black

    @property
    def board(self) -> chess.Board:
        if self.initial_fen != "startpos":
            raise ValueError("Custom starting positions are not yet supported")
        board = chess.Board()
        for uci in self.moves:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"Illegal move in Lichess game state: {uci}")
            board.push(move)
        return board

    @property
    def fen(self) -> str:
        return self.board.fen()

    @property
    def ply(self) -> int:
        return len(self.moves)

    @property
    def is_bot_turn(self) -> bool:
        return self.board.turn == self.bot_color

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_playable(self) -> bool:
        return self.status == "started"

    @property
    def time_control(self) -> ExactTimeControl | None:
        if self.initial_clock_ms is None or self.increment_ms is None:
            return None
        return time_control_from_clock(self.initial_clock_ms, self.increment_ms)

    def apply_game_state(self, event: dict[str, tp.Any]) -> None:
        """Apply an immutable full snapshot from Lichess.

        The API sends the complete UCI move string on every update.  Replacing
        the list makes reconnects and duplicate events idempotent.
        """

        previous_moves = self.moves
        previous_wtime_ms = self.wtime_ms
        previous_btime_ms = self.btime_ms
        previous_winc_ms = self.winc_ms
        previous_binc_ms = self.binc_ms
        moves = event.get("moves", "")
        new_moves = str(moves).split() if moves else []
        self.moves = new_moves
        self.wtime_ms = _optional_int(event.get("wtime"))
        self.btime_ms = _optional_int(event.get("btime"))
        self.winc_ms = _optional_int(event.get("winc"))
        self.binc_ms = _optional_int(event.get("binc"))
        self.status = str(event.get("status", self.status))
        self._update_move_seconds(
            previous_moves,
            previous_wtime_ms,
            previous_btime_ms,
            previous_winc_ms,
            previous_binc_ms,
        )
        self._update_clock_histories(previous_moves)

    def _update_move_seconds(
        self,
        previous_moves: list[str],
        previous_wtime_ms: int | None,
        previous_btime_ms: int | None,
        previous_winc_ms: int | None,
        previous_binc_ms: int | None,
    ) -> None:
        if len(self.move_seconds) > len(self.moves):
            self.move_seconds = self.move_seconds[: len(self.moves)]
        if len(self.move_seconds) < len(self.moves):
            self.move_seconds.extend([0] * (len(self.moves) - len(self.move_seconds)))
        if len(self.moves) != len(previous_moves) + 1:
            return
        if previous_wtime_ms is None or previous_btime_ms is None:
            return
        moving_white = len(previous_moves) % 2 == 0
        previous_clock = previous_wtime_ms if moving_white else previous_btime_ms
        current_clock = self.wtime_ms if moving_white else self.btime_ms
        increment = previous_winc_ms if moving_white else previous_binc_ms
        if current_clock is None:
            return
        if increment is None:
            increment = self.increment_ms or 0
        elapsed_ms = previous_clock + increment - current_clock
        self.move_seconds[-1] = max(0, int(round(elapsed_ms / 1000)))

    def _update_clock_histories(self, previous_moves: list[str]) -> None:
        if self.wtime_ms is None or self.btime_ms is None:
            return

        # BoardHistory indexes clocks by each player's move number, rather than
        # by every ply.  Pad moves that predate this process with ``None`` so a
        # reconnect snapshot cannot shift the observed clocks to earlier moves.
        white_count = (len(self.moves) + 1) // 2
        black_count = len(self.moves) // 2
        self.white_clock_history = self.white_clock_history[:white_count]
        self.black_clock_history = self.black_clock_history[:black_count]
        self.white_clock_history.extend(
            [None] * (white_count - len(self.white_clock_history))
        )
        self.black_clock_history.extend(
            [None] * (black_count - len(self.black_clock_history))
        )

        move_delta = len(self.moves) - len(previous_moves)
        if move_delta == 1:
            moving_white = len(previous_moves) % 2 == 0
            if moving_white and white_count:
                self.white_clock_history[-1] = self.wtime_ms / 1000
            elif black_count:
                self.black_clock_history[-1] = self.btime_ms / 1000
            return

        # A full/reconnect snapshot provides only the latest clock for each
        # side.  Preserve its correct index and leave unavailable history empty.
        if white_count:
            self.white_clock_history[-1] = self.wtime_ms / 1000
        if black_count:
            self.black_clock_history[-1] = self.btime_ms / 1000

    def to_pgn(self) -> str:
        if self.initial_fen != "startpos":
            raise ValueError("Custom starting positions are not yet supported")
        game = chess.pgn.Game()
        game.headers["White"] = self.white.name
        game.headers["Black"] = self.black.name
        if self.white.rating is not None:
            game.headers["WhiteElo"] = str(self.white.rating)
        if self.black.rating is not None:
            game.headers["BlackElo"] = str(self.black.rating)
        if self.time_control is not None:
            game.headers["TimeControl"] = self.time_control.value
        node = game
        board = chess.Board()
        for uci in self.moves:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"Illegal move in Lichess game state: {uci}")
            node = node.add_variation(move)
            board.push(move)
        exporter = chess.pgn.StringExporter(
            headers=True, variations=False, comments=False
        )
        return game.accept(exporter)


@dataclasses.dataclass(frozen=True)
class MoveDecision:
    move_uci: str
    probabilities: dict[chess.Move, float] | None = None
    resigned: bool = False
    offered_draw: bool = False


def _player_info(data: dict[str, tp.Any]) -> PlayerInfo:
    return PlayerInfo(
        id=data.get("id"),
        name=str(data.get("name") or data.get("id") or "?"),
        rating=_optional_int(data.get("rating")),
    )


def _optional_int(value: tp.Any) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = ["LiveGameState", "MoveDecision", "PlayerInfo"]
