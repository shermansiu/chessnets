# Lichess bot runner

The optional `lichess-bot` extra provides a terminal and notebook interface for
running an existing chessnets pipeline against Lichess:

```bash
pip install "chessnets[lichess-bot]"
export LICHESS_TOKEN="..."
chessnets lichess-bot validate --config examples/lichess-game-aware.yaml
chessnets lichess-bot run --config examples/lichess-game-aware.yaml
```

This is ChessNets' own runner for the Lichess Bot API, not the separate
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) project. If you
prefer that project, configure `chessnets-uci` as its UCI engine instead; see
[the UCI engine guide](uci-engine.md).

The token must have the Lichess `bot:play` permission and belong to a Bot
account. Keep it in the environment rather than in YAML.

There are two metadata modes:

* `frozen` uses the values in `metadata.values` for every game. The live
  position and move history still change. This is useful for reproducing a
  fixed inference condition.
* `game` derives ratings and the actual time control from the Lichess game,
  using `metadata.fallbacks` when a value is unavailable. Static prompt values
  can still be placed in `metadata.values`.

The runner currently accepts standard-chess games, uses one active game by
default, and validates every returned move against the local board before
submitting it.

Notebook usage does not start a nested event loop:

```python
import os

from chessnets.play import LichessBotConfig, LichessBotRunner

config = LichessBotConfig.from_yaml("examples/lichess-game-aware.yaml")
runner = LichessBotRunner.from_config(
    config,
    token=os.environ["LICHESS_TOKEN"],
)
await runner.run_async()
```

For a background task, use `asyncio.create_task(runner.run_async())` and later
`await runner.stop()`.
