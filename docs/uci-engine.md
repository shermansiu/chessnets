# UCI engine

The `chessnets-uci` command exposes a pipeline as a UCI engine for chess GUIs
such as Cute Chess, Arena, and Banksia:

```bash
pip install chessnets
chessnets-uci --config examples/uci-frozen.yaml
```

`chessnets-uci` is a standalone alias for `chessnets uci`, so GUIs can point
directly at one executable. `run` is the default command, so GUIs can launch
the engine with just `--config`. Use `chessnets-uci validate --config ...` to check a configuration,
including any `--model-id` or `--device` overrides, without loading a model.

The process uses stdout exclusively for UCI responses. Logs and diagnostics go
to stderr, so the command can be configured directly as an engine executable.

UCI supplies a position and optional current clocks, but not dependable player
ratings, prompts, or the original time control. The engine therefore requires
`metadata.mode: frozen`; values in `metadata.values` are used for every
position. The side to move is treated as the engine side when orienting
`bot_elo` and `opponent_elo`.

The implementation performs one pipeline prediction per `go` command. It
accepts standard timing fields (`wtime`, `btime`, `winc`, `binc`) as context,
but does not expose depth-, node-, or score-based search options. The
`searchmoves` restriction is honored when the GUI supplies it. Other
unsupported commands are ignored or reported with `info string` while
preserving the UCI session.

Notebook callers can use the direct async API without a nested event loop:

```python
import chess
from chessnets.play import PipelineUciEngine, UciEngineConfig

config = UciEngineConfig.from_yaml("examples/uci-frozen.yaml")
engine = PipelineUciEngine.from_config(config)
decision = await engine.play_async(chess.Board())
print(decision.move_uci)
```

For synchronous scripts outside an event loop, use `engine.play(board)`.
