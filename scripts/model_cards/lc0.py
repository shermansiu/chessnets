"""Generated section of Lc0 and Maia-1 model cards."""

from __future__ import annotations

import pathlib

from model_cards import common

GENERATED_HEADING = "## Use with ChessNets"

_MAIA1_DESCRIPTION = (
    "This Maia-1 network is intended to imitate moves from its target human "
    "rating group. It is a move-prediction model, not a calibrated estimate "
    "of an individual player's strength."
)
_LC0_DESCRIPTION = (
    "This is an Lc0 policy/value network. Its playing behavior and "
    "strength depend on the source network and the inference/search settings."
)

_MAIA1_CITATION = """## Citation

Please cite the original Maia work:

```bibtex
@inproceedings{mcilroyyoung2020maia,
  title={Aligning Superhuman AI with Human Behavior: Chess as a Model System},
  author={McIlroy-Young, Reid and Sen, Siddhartha and Kleinberg, Jon and Anderson, Ashton},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2020},
  doi={10.1145/3394486.3403219}
}
```

See the [official Maia repository](https://github.com/CSSLab/maia-chess) and
[paper](https://arxiv.org/abs/2006.01855)."""

_LC0_CITATION = """## Citation and attribution

Credit the source network identified above and the
[Leela Chess Zero project](https://lczero.org/). This community checkpoint does
not have a model-specific paper citation until the official project supplies one."""


def generated_section(repository_id: str, *, maia1: bool) -> str:
    return f"""{GENERATED_HEADING}

```python
from chessnets.pipelines.lc0 import Lc0Pipeline

pipeline = Lc0Pipeline.from_pretrained("{repository_id}")
predictions = pipeline("1. e4 e5")
```

## Checkpoint artifacts

- `config.json`: canonical, fully materialized ChessNets architecture config
- `model.safetensors`: decoded float32 PyTorch tensors
- `provenance.json`: source and conversion hashes
- `original/model.pb.gz`: original Lc0 protobuf checkpoint
- `training/original_lczero_training_config.yaml`: original training config

The learned protobuf weight layers used `LINEAR16` encoding. Conversion decoded
those values into float32 tensors; it cannot restore precision already lost in
the source encoding.

## Intended use and limitations

{_MAIA1_DESCRIPTION if maia1 else _LC0_DESCRIPTION}

Review legal moves and probabilities in context. Outputs are not guarantees of
playing strength, fairness, or suitability for high-stakes decisions.

{_MAIA1_CITATION if maia1 else _LC0_CITATION}"""


def update(root: pathlib.Path, *, write: bool, repository_id: str, maia1: bool) -> bool:
    current = common.read_readme(root, required=True)
    updated = common.replace_tail(
        current, GENERATED_HEADING, generated_section(repository_id, maia1=maia1)
    )
    return common.write_or_preview(root, current, updated, write=write)
