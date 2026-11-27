"""UCI protocol support for chessnets pipelines.

The UCI layer is intentionally small: a pipeline supplies one move for one
position, while the protocol driver deals with parsing, state isolation, and
the stdin/stdout contract expected by chess GUIs.  In particular, protocol
output is never mixed with logging or model progress output.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import pathlib
import sys
import typing as tp

import chess
import chess.pgn

from .adapters import PLAY_ADAPTERS, PipelineAdapterError, PipelinePlayAdapter
from .config import MetadataConfig, UciEngineConfig
from .context import MoveDecision, PlayerInfo
from .matching import ExactTimeControl
from .metadata import FrozenMetadataResolver, MetadataResolutionError, ResolvedMetadata

logger = logging.getLogger(__name__)


class UciProtocolError(ValueError):
    """Raised for malformed UCI commands or positions."""


@dataclasses.dataclass
class UciPosition:
    """A position reconstructed from UCI ``position`` commands.

    ``engine_color`` is set to the side to move for each ``go`` command.  UCI
    has no separate player identity, so this is the only unambiguous way to
    orient frozen ``bot_elo``/``opponent_elo`` metadata.
    """

    board: chess.Board
    moves: list[str] = dataclasses.field(default_factory=list)
    initial_fen: str = "startpos"
    engine_color: chess.Color = chess.WHITE
    wtime_ms: int | None = None
    btime_ms: int | None = None
    winc_ms: int | None = None
    binc_ms: int | None = None
    white_elo: int | None = None
    black_elo: int | None = None
    pgn_time_control: str | None = None
    white_clock_history: list[float | None] = dataclasses.field(default_factory=list)
    black_clock_history: list[float | None] = dataclasses.field(default_factory=list)
    move_seconds: list[int] = dataclasses.field(default_factory=list)

    @classmethod
    def startpos(cls, moves: tp.Iterable[str] = ()) -> UciPosition:
        board = chess.Board()
        result = cls(board=board, initial_fen="startpos")
        result._push_moves(moves)
        return result

    @classmethod
    def from_fen(cls, fen: str, moves: tp.Iterable[str] = ()) -> UciPosition:
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise UciProtocolError(f"Invalid FEN: {fen}") from exc
        result = cls(board=board, initial_fen=fen)
        result._push_moves(moves)
        return result

    def copy(self) -> UciPosition:
        return dataclasses.replace(
            self,
            board=self.board.copy(stack=False),
            moves=list(self.moves),
            white_clock_history=list(self.white_clock_history),
            black_clock_history=list(self.black_clock_history),
            move_seconds=list(self.move_seconds),
        )

    @property
    def fen(self) -> str:
        return self.board.fen()

    @property
    def ply(self) -> int:
        return len(self.moves)

    @property
    def bot_color(self) -> chess.Color:
        return self.engine_color

    @property
    def bot_username(self) -> str:
        return "engine"

    @property
    def white(self) -> PlayerInfo:
        return PlayerInfo(id="white", name="White")

    @property
    def black(self) -> PlayerInfo:
        return PlayerInfo(id="black", name="Black")

    @property
    def bot_player(self) -> PlayerInfo:
        return self.white if self.engine_color == chess.WHITE else self.black

    @property
    def opponent(self) -> PlayerInfo:
        return self.black if self.engine_color == chess.WHITE else self.white

    @property
    def is_bot_turn(self) -> bool:
        return self.board.turn == self.engine_color

    @property
    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=False)

    @property
    def is_playable(self) -> bool:
        return not self.is_terminal

    @property
    def time_control(self) -> ExactTimeControl | None:
        # ``go wtime/btime`` contains the remaining clocks, not the original
        # game control.  Treating those values as an initial time control can
        # silently select the wrong model token, so UCI metadata never derives
        # a time control from the current position.
        return None

    def to_pgn(self) -> str:
        game = chess.pgn.Game()
        game.headers["White"] = self.white.name
        game.headers["Black"] = self.black.name
        if self.white_elo is not None:
            game.headers["WhiteElo"] = str(self.white_elo)
        if self.black_elo is not None:
            game.headers["BlackElo"] = str(self.black_elo)
        if self.pgn_time_control is not None:
            game.headers["TimeControl"] = str(self.pgn_time_control)
        if self.initial_fen != "startpos":
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = self.initial_fen
        node = game
        board = (
            chess.Board()
            if self.initial_fen == "startpos"
            else chess.Board(self.initial_fen)
        )
        for move_uci in self.moves:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                raise UciProtocolError(f"Illegal move in position: {move_uci}")
            node = node.add_variation(move)
            board.push(move)
        exporter = chess.pgn.StringExporter(
            headers=True, variations=False, comments=False
        )
        return game.accept(exporter)

    def _push_moves(self, moves: tp.Iterable[str]) -> None:
        for move_uci in moves:
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError as exc:
                raise UciProtocolError(f"Invalid UCI move: {move_uci}") from exc
            if move not in self.board.legal_moves:
                raise UciProtocolError(f"Illegal move in position: {move_uci}")
            self.board.push(move)
            self.moves.append(move_uci)


@dataclasses.dataclass(frozen=True)
class UciGoParameters:
    """Parsed timing/search fields from a UCI ``go`` command."""

    wtime: int | None = None
    btime: int | None = None
    winc: int | None = None
    binc: int | None = None
    movestogo: int | None = None
    depth: int | None = None
    nodes: int | None = None
    mate: int | None = None
    infinite: bool = False
    ponder: bool = False
    searchmoves: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "wtime",
            "btime",
            "winc",
            "binc",
            "movestogo",
            "depth",
            "nodes",
            "mate",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"go {name} must be non-negative")


class PipelineUciEngine:
    """Notebook-friendly adapter that runs one pipeline prediction per move."""

    def __init__(
        self,
        pipeline: tp.Any,
        config: UciEngineConfig | None = None,
        *,
        metadata: MetadataConfig | None = None,
        adapter: PipelinePlayAdapter | None = None,
    ) -> None:
        if config is None:
            config = UciEngineConfig(
                model_id=type(pipeline).__name__,
                metadata=metadata or MetadataConfig(),
            )
        self.pipeline = pipeline
        self.config = config
        self.adapter = adapter or PLAY_ADAPTERS.get(pipeline)
        self.resolver = FrozenMetadataResolver(config.metadata)
        self._inference_lock = asyncio.Lock()
        self._validate_configuration()

    @classmethod
    def from_config(
        cls,
        config: UciEngineConfig,
        *,
        pipeline: tp.Any = None,
    ) -> PipelineUciEngine:
        if pipeline is None:
            from ..pipelines.auto_pipeline import AutoPipeline

            pipeline = AutoPipeline.from_pretrained(
                config.model_id, device=config.device, **config.model_kwargs
            )
        return cls(pipeline, config)

    async def play_async(
        self,
        position: UciPosition | chess.Board,
        go: UciGoParameters | None = None,
    ) -> MoveDecision:
        state = _coerce_position(position, go)
        state.engine_color = state.board.turn
        supported = _supported_time_controls(self.pipeline)
        metadata = self.resolver.resolve(state, supported_time_controls=supported)
        _apply_pgn_metadata(state, metadata.values)
        # Current UCI clocks are dynamic position context rather than frozen
        # model metadata.  Expose a one-sample history for pipelines that
        # explicitly request clock inputs.
        values = dict(metadata.values)
        if state.white_clock_history and state.black_clock_history:
            values.setdefault("white_clock_times", state.white_clock_history)
            values.setdefault("black_clock_times", state.black_clock_history)
            values.setdefault(
                "clock_times",
                (state.white_clock_history, state.black_clock_history),
            )
        metadata = ResolvedMetadata(
            values,
            actual_time_control=metadata.actual_time_control,
            model_time_control=metadata.model_time_control,
        )
        self.adapter.validate_launch(self.pipeline, metadata)
        async with self._inference_lock:
            result = self.adapter.predict(self.pipeline, state, metadata)
            if self.config.inference_timeout_seconds is not None:
                result = await asyncio.wait_for(
                    result, timeout=self.config.inference_timeout_seconds
                )
            else:
                result = await result
        return result

    def play(
        self,
        position: UciPosition | chess.Board,
        go: UciGoParameters | None = None,
    ) -> MoveDecision:
        """Synchronously predict a move for scripts outside an event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.play_async(position, go))
        raise RuntimeError(
            "PipelineUciEngine.play() cannot run inside an active event loop; "
            "use `await engine.play_async(...)` in a notebook."
        )

    def _validate_configuration(self) -> None:
        state = UciPosition.startpos()
        supported = _supported_time_controls(self.pipeline)
        metadata = self.resolver.resolve(state, supported_time_controls=supported)
        values = dict(metadata.values)
        # Clock values arrive with ``go`` and cannot be known at process
        # startup.  Mark those explicitly dynamic while still validating all
        # genuinely frozen inputs (ratings, prompts, time-control tokens, ...).
        dynamic = {
            "white_clock_times",
            "black_clock_times",
            "clock_times",
            "move_seconds",
        }
        for name in self.pipeline.input_config.required_inputs & dynamic:
            values.setdefault(name, [0.0])
        metadata = ResolvedMetadata(
            values,
            actual_time_control=metadata.actual_time_control,
            model_time_control=metadata.model_time_control,
        )
        self.adapter.validate_launch(self.pipeline, metadata)


class UciProtocol:
    """Asynchronous UCI command processor.

    ``handle_line`` is useful in tests and notebook integrations.  ``run``
    connects it to text streams and writes protocol lines with flushing.
    """

    def __init__(
        self,
        engine: PipelineUciEngine | tp.Any,
        *,
        output: tp.Callable[[str], None] | None = None,
    ) -> None:
        self.engine = (
            engine
            if isinstance(engine, PipelineUciEngine)
            else PipelineUciEngine(engine)
        )
        self.position = UciPosition.startpos()
        self.output = output
        self.outputs: list[str] = []
        self._search_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._quit = False

    async def handle_line(self, line: str) -> list[str]:
        """Handle one command and return lines emitted synchronously."""

        start = len(self.outputs)
        text = line.strip()
        if not text:
            return []
        try:
            words = text.split()
            command = words[0].lower()
            if command == "uci":
                self.emit(f"id name {self.engine.config.name}")
                self.emit(f"id author {self.engine.config.author}")
                self.emit("uciok")
            elif command == "isready":
                self.emit("readyok")
            elif command == "ucinewgame":
                await self._invalidate_search(emit_fallback=False)
                self.position = UciPosition.startpos()
            elif command == "position":
                await self._invalidate_search(emit_fallback=False)
                self.position = _parse_position(words)
            elif command == "go":
                await self._start_search(_parse_go(words[1:]))
            elif command == "stop":
                await self._invalidate_search(emit_fallback=True)
            elif command == "quit":
                await self._invalidate_search(emit_fallback=False)
                self._quit = True
            elif command in {"debug", "register", "ponderhit", "setoption"}:
                # No options are advertised yet.  Consuming these commands is
                # preferable to terminating a GUI session on harmless setup.
                if command == "setoption":
                    _parse_setoption(words[1:])
            else:
                self.emit(f"info string Unknown command: {words[0]}")
        except (
            UciProtocolError,
            PipelineAdapterError,
            MetadataResolutionError,
            ValueError,
        ) as exc:
            self.emit(f"info string {exc}")
        return self.outputs[start:]

    async def run(
        self,
        reader: tp.TextIO | None = None,
        writer: tp.TextIO | None = None,
    ) -> None:
        """Run the protocol on blocking text streams until EOF or ``quit``."""

        reader = sys.stdin if reader is None else reader
        writer = sys.stdout if writer is None else writer

        def write(line: str) -> None:
            writer.write(line + "\n")
            writer.flush()

        previous = self.output
        self.output = write
        try:
            while not self._quit:
                line = await asyncio.to_thread(reader.readline)
                if not line:
                    break
                await self.handle_line(line)
            if self._search_task is not None:
                await self._search_task
        finally:
            self.output = previous

    async def run_async(
        self,
        reader: tp.TextIO | None = None,
        writer: tp.TextIO | None = None,
    ) -> None:
        """Alias matching the async naming used by the Lichess runner."""

        await self.run(reader, writer)

    def emit(self, line: str) -> None:
        self.outputs.append(line)
        if self.output is not None:
            self.output(line)

    async def _start_search(self, go: UciGoParameters) -> None:
        await self._invalidate_search(emit_fallback=False)
        self._generation += 1
        generation = self._generation
        position = self.position.copy()
        self._search_task = asyncio.create_task(
            self._search(position, go, generation), name="chessnets-uci-search"
        )

    async def _search(
        self, position: UciPosition, go: UciGoParameters, generation: int
    ) -> None:
        try:
            decision = await self.engine.play_async(position, go)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            if generation == self._generation and not self._quit:
                self.emit(f"info string inference failed: {exc}")
                self.emit(f"bestmove {_fallback_move(position.board)}")
            return
        if go.searchmoves and decision.move_uci not in go.searchmoves:
            allowed = _legal_searchmoves(position.board, go.searchmoves)
            decision = dataclasses.replace(
                decision, move_uci=allowed[0] if allowed else "0000"
            )
        if generation == self._generation and not self._quit:
            self.emit(f"bestmove {decision.move_uci}")

    async def _invalidate_search(self, *, emit_fallback: bool) -> None:
        self._generation += 1
        task = self._search_task
        self._search_task = None
        if emit_fallback and task is not None and not task.done():
            self.emit(f"bestmove {_fallback_move(self.position.board)}")


# Short aliases make the public API discoverable without forcing callers to
# know whether they want the protocol driver or the direct notebook adapter.
UciEngine = PipelineUciEngine
UciServer = UciProtocol
UciConfig = UciEngineConfig


def _coerce_position(
    position: UciPosition | chess.Board, go: UciGoParameters | None
) -> UciPosition:
    if isinstance(position, UciPosition):
        state = position.copy()
    elif isinstance(position, chess.Board):
        state = UciPosition(
            board=position.copy(stack=False), initial_fen=position.fen()
        )
    else:
        raise TypeError("position must be a UciPosition or chess.Board")
    if go is not None:
        state.wtime_ms = go.wtime
        state.btime_ms = go.btime
        state.winc_ms = go.winc
        state.binc_ms = go.binc
        if go.wtime is not None:
            state.white_clock_history = [go.wtime / 1000]
        if go.btime is not None:
            state.black_clock_history = [go.btime / 1000]
    return state


def _apply_pgn_metadata(state: UciPosition, values: dict[str, tp.Any]) -> None:
    state.white_elo = _metadata_scalar(values.get("white_elos"))
    state.black_elo = _metadata_scalar(values.get("black_elos"))
    time_control = values.get("time_controls")
    state.pgn_time_control = (
        str(_metadata_scalar(time_control)) if time_control is not None else None
    )


def _metadata_scalar(value: tp.Any) -> tp.Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _parse_position(words: list[str] | str) -> UciPosition:
    if isinstance(words, str):
        words = words.split()
    if words and words[0].lower() != "position":
        words = ["position", *words]
    if len(words) < 2:
        raise UciProtocolError("position requires startpos or fen")
    kind = words[1].lower()
    if kind == "startpos":
        move_start = 2
        initial = UciPosition.startpos()
    elif kind == "fen":
        if len(words) < 8:
            raise UciProtocolError("position fen requires six FEN fields")
        move_start = next(
            (
                index
                for index in range(2, len(words))
                if words[index].lower() == "moves"
            ),
            len(words),
        )
        if move_start == len(words):
            move_start = len(words)
        fen_fields = words[2:move_start]
        if len(fen_fields) != 6:
            raise UciProtocolError("position fen requires six FEN fields")
        initial = UciPosition.from_fen(" ".join(fen_fields))
    else:
        raise UciProtocolError(f"Unknown position form: {words[1]}")
    if move_start < len(words):
        if words[move_start].lower() != "moves":
            raise UciProtocolError("Expected 'moves' in position command")
        initial._push_moves(words[move_start + 1 :])
    return initial


def _parse_go(words: list[str] | str) -> UciGoParameters:
    if isinstance(words, str):
        words = words.split()
    if words and words[0].lower() == "go":
        words = words[1:]
    values: dict[str, tp.Any] = {}
    searchmoves: list[str] = []
    i = 0
    value_keys = {
        "wtime",
        "btime",
        "winc",
        "binc",
        "movestogo",
        "depth",
        "nodes",
        "mate",
    }
    while i < len(words):
        key = words[i].lower()
        if key in {"infinite", "ponder"}:
            values[key] = True
            i += 1
        elif key in value_keys:
            if i + 1 >= len(words):
                raise UciProtocolError(f"Missing value for go {key}")
            try:
                values[key] = int(words[i + 1])
            except ValueError as exc:
                raise UciProtocolError(f"Invalid value for go {key}") from exc
            i += 2
        elif key == "searchmoves":
            searchmoves = words[i + 1 :]
            break
        else:
            # UCI GUIs sometimes send newer fields.  They are harmless for a
            # one-shot pipeline, so preserve forward compatibility here.
            i += 1
    values["searchmoves"] = tuple(searchmoves)
    return UciGoParameters(**values)


def _parse_setoption(words: list[str]) -> tuple[str, str]:
    if not words or words[0].lower() != "name":
        raise UciProtocolError("setoption requires name")
    try:
        value_start = words.index("value", 1)
    except ValueError:
        value_start = len(words)
    name = " ".join(words[1:value_start]).strip()
    if not name:
        raise UciProtocolError("setoption requires a non-empty name")
    return name, " ".join(words[value_start + 1 :]) if value_start < len(words) else ""


def _fallback_move(board: chess.Board) -> str:
    return next((move.uci() for move in board.legal_moves), "0000")


def _legal_searchmoves(board: chess.Board, moves: tp.Iterable[str]) -> list[str]:
    legal = {move.uci() for move in board.legal_moves}
    return [move for move in moves if move in legal]


def _supported_time_controls(pipeline: tp.Any) -> tuple[str, ...]:
    tokenizer = getattr(pipeline, "tokenizer", None)
    tokens = getattr(tokenizer, "tokens", None)
    if tokens is None:
        return ()
    return tuple(token for token in tokens if isinstance(token, str) and "+" in token)


parse_position = _parse_position
parse_go = _parse_go


async def run_uci(
    engine: PipelineUciEngine,
    reader: tp.TextIO | None = None,
    writer: tp.TextIO | None = None,
) -> None:
    """Run a UCI server using the supplied engine and text streams."""

    protocol = UciProtocol(engine)
    await protocol.run(
        sys.stdin if reader is None else reader,
        sys.stdout if writer is None else writer,
    )


def build_parser(prog: str = "chessnets-uci") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run a chessnets pipeline as a UCI chess engine.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "validate"),
        default="run",
        help=(
            "run the engine over stdin/stdout (default), or validate the "
            "configuration without loading a model"
        ),
    )
    parser.add_argument(
        "--config", required=True, type=pathlib.Path, help="YAML engine configuration"
    )
    parser.add_argument("--model-id", help="override the configured model ID")
    parser.add_argument("--device", help="override the configured device")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="logging level for stderr (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "chessnets-uci") -> int:
    args = build_parser(prog).parse_args(argv)
    config = UciEngineConfig.from_yaml(args.config)
    if args.model_id or args.device:
        config = dataclasses.replace(
            config,
            model_id=args.model_id or config.model_id,
            device=args.device or config.device,
        )
    if args.command == "validate":
        # Validation without loading a model checks only the config contract.
        print(f"Configuration is valid for model {config.model_id!r}.")
        return 0
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    engine = PipelineUciEngine.from_config(config)
    protocol = UciProtocol(engine)
    asyncio.run(protocol.run(sys.stdin, sys.stdout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PipelineUciEngine",
    "UciEngine",
    "UciEngineConfig",
    "UciConfig",
    "UciGoParameters",
    "UciPosition",
    "UciProtocol",
    "UciProtocolError",
    "UciServer",
    "build_parser",
    "main",
    "parse_go",
    "parse_position",
    "run_uci",
]
