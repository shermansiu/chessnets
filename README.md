# ChessNets

[![CI](https://github.com/shermansiu/chessnets/actions/workflows/ci.yml/badge.svg)](https://github.com/shermansiu/chessnets/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/chessnets)](https://pypi.org/project/chessnets/)
[![Python versions](https://img.shields.io/pypi/pyversions/chessnets)](https://pypi.org/project/chessnets/)
[![Downloads](https://static.pepy.tech/badge/chessnets/month)](https://pepy.tech/project/chessnets)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/shermansiu/chessnets/blob/main/LICENSE)

State-of-the-art models for playing chess and imitating human moves, behind
one PyTorch API.

ChessNets loads Lc0, Maia, Maia 2, UniMaia, Allie, ChessGPT, and ChessCLIP
checkpoints from the Hugging Face Hub and turns games into move predictions.
It can also run any of them as a UCI engine for chess GUIs or as a Lichess bot.

## Installation

```shell
pip install chessnets
```

The core install supports Lc0 and Maia-1 networks and the UCI engine. Other
model families need an optional extra:

| Extra | Adds |
|---|---|
| `maia2` | Maia 2 |
| `unimaia` | UniMaia |
| `allie` | Allie |
| `text-generation` | ChessGPT, Carlini-style prompting, and OpenAI-compatible endpoints |
| `chess-clip` | ChessCLIP |
| `lichess-bot` | The Lichess Bot API runner |
| `notebook` | Board rendering in Jupyter |

For example, `pip install "chessnets[maia2,lichess-bot]"`. If an extra is
missing, the error message names the one to install.

## Quickstart

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("maia-1500", device="cpu")
predictions = pipeline.make_prediction("1. e4 e5 2. Nf3")
print(predictions.best_move_sans)  # ['Nc6']
```

`make_prediction` accepts one PGN or a list of PGNs and batches them
automatically. `AutoPipeline` also accepts full Hugging Face repository IDs,
local directories, and any repository whose `config.json` names a supported
architecture. See [Loading models](https://github.com/shermansiu/chessnets/blob/main/docs/loading-models.md).

## Supported models

Each family's page covers its origin, usage and citation.

| Family | Example names | Extra |
|---|---|---|
| [Maia-1](https://github.com/shermansiu/chessnets/blob/main/docs/models/maia1.md) | `maia-1100` … `maia-1900`, `maia-2200` | — |
| [Lc0 networks](https://github.com/shermansiu/chessnets/blob/main/docs/models/lc0.md) | `lc0_BT4-1024x15x32h-swa-6147500`, `lc0_t82-768x15x24h-swa-7364000` | — |
| [Maia 2](https://github.com/shermansiu/chessnets/blob/main/docs/models/maia2.md) | `maia2-blitz`, `maia2-rapid` | `maia2` |
| [UniMaia](https://github.com/shermansiu/chessnets/blob/main/docs/models/unimaia.md) | `unimaia`, `unimaia-aux` | `unimaia` |
| [Allie](https://github.com/shermansiu/chessnets/blob/main/docs/models/allie.md) | `allie_medium_best` | `allie` |
| [ChessGPT](https://github.com/shermansiu/chessnets/blob/main/docs/models/chessgpt.md) | `Waterhorse/chessgpt-base-v1`, `Waterhorse/chessgpt-chat-v1_simple` | `text-generation` |
| [ChessCLIP](https://github.com/shermansiu/chessnets/blob/main/docs/models/chess-clip.md) | `Waterhorse/ChessCLIP` | `chess-clip` |

ChessGPT profiles constrain generation with a regular expression by default,
which keeps output in move notation. To restrict generation to the legal moves
in each position, use multiple-choice mode:

```python
pipeline = AutoPipeline.from_pretrained(
    "Waterhorse/chessgpt-base-v1", device="cuda", constraint="choice"
)
```

## Playing

```shell
chessnets uci --config examples/uci-frozen.yaml           # UCI engine for chess GUIs
chessnets lichess-bot run --config examples/lichess-game-aware.yaml
```

The [example configurations](https://github.com/shermansiu/chessnets/tree/main/examples)
use Maia 2, so install `chessnets[maia2]` (plus `lichess-bot` for the bot).
Run `chessnets --help` for every command. The UCI engine is also installed as
`chessnets-uci`, so GUIs can launch it directly.

## Documentation

- [Model families](https://github.com/shermansiu/chessnets/tree/main/docs/models)
- [Loading models](https://github.com/shermansiu/chessnets/blob/main/docs/loading-models.md)
- [UCI engine](https://github.com/shermansiu/chessnets/blob/main/docs/uci-engine.md)
- [Lichess bot](https://github.com/shermansiu/chessnets/blob/main/docs/lichess-bot.md)
- [Checkpoint format and Lc0 conversion](https://github.com/shermansiu/chessnets/blob/main/docs/checkpoint-format.md)
- [Working with board histories](https://github.com/shermansiu/chessnets/blob/main/docs/board-history.md)
- [Efficient inference for small requests](https://github.com/shermansiu/chessnets/blob/main/docs/inference-performance.md)
- [Contributing models and pipelines](https://github.com/shermansiu/chessnets/blob/main/CONTRIBUTING.md)

## License

ChessNets is licensed under the [Apache License 2.0](https://github.com/shermansiu/chessnets/blob/main/LICENSE).
Some model implementations are adapted from MIT-licensed projects and keep
their original licenses; see
[`THIRD_PARTY_NOTICES.md`](https://github.com/shermansiu/chessnets/blob/main/THIRD_PARTY_NOTICES.md).
Model weights have their own licenses, stated on each model card.

## Citation

If you use ChessNets in your research, please cite it:

```bibtex
@software{siu2026chessnets,
  author  = {Siu, Sherman},
  title   = {{ChessNets}: State-of-the-art models for playing chess and human move imitation},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/shermansiu/chessnets},
  license = {Apache-2.0}
}
```

The same metadata is in
[`CITATION.cff`](https://github.com/shermansiu/chessnets/blob/main/CITATION.cff),
which GitHub uses for its "Cite this repository" button.

If you use a specific model, please also cite its authors. Each
[model page](https://github.com/shermansiu/chessnets/tree/main/docs/models)
gives the citation to use.
