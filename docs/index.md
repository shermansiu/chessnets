# ChessNets

State-of-the-art models for playing chess and imitating human moves, behind
one PyTorch API.

ChessNets loads Lc0, Maia, Maia 2, UniMaia, Allie, ChessGPT, and ChessCLIP
checkpoints from the Hugging Face Hub and turns games into move predictions.
It can also run any of them as a UCI engine for chess GUIs or as a Lichess bot.

```shell
pip install chessnets
```

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("maia-1500", device="cpu")
predictions = pipeline.make_prediction("1. e4 e5 2. Nf3")
print(predictions.best_move_sans)  # ['Nc6']
```

- **Get started:** [Loading models](loading-models.md) explains names,
  repository IDs, local directories and devices.
- **Pick a model:** each [model family](models/README.md) has a page with its
  install extra, loading call and citation.
- **Play:** run any model as a [UCI engine](uci-engine.md) or a
  [Lichess bot](lichess-bot.md).

Optional extras, citation details and the license are in the
[README on GitHub](https://github.com/shermansiu/chessnets#readme).
