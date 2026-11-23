"""Generated section of Lc0 and Maia-1 model cards."""

from __future__ import annotations

import json
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

_LC0_ATTRIBUTION = """## Citation and attribution

Credit the source network identified above and the
[Leela Chess Zero project](https://lczero.org/)."""

_LC0_SE_CITATION = (
    _LC0_ATTRIBUTION
    + """ This SE-ResNet network has no
paper of its own."""
)

# Lc0's transformer networks (BTLeelaZeroNet in ChessNets) use the Chessformer
# architecture.
_LC0_TRANSFORMER_CITATION = (
    _LC0_ATTRIBUTION
    + """ This transformer network uses
the Chessformer architecture; please also cite:

```bibtex
@misc{monroe2024mastering,
  title         = {Mastering Chess with a Transformer Model},
  author        = {Monroe, Daniel and Chalmers, Philip A.},
  year          = {2024},
  eprint        = {2409.12272},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2409.12272}
}

@inproceedings{monroe2026chessformer,
  title     = {{Chessformer}: A Unified Architecture for Chess Modeling},
  author    = {Monroe, Daniel and Eilender, George and Chalmers, Philip and Tang, Zhenwei and Anderson, Ashton},
  booktitle = {International Conference on Learning Representations},
  editor    = {C. Vondrick and B. Hariharan and C. Raffel and L. Pinto and D. Yang and A. Faust},
  pages     = {36245--36271},
  volume    = {2026},
  year      = {2026},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2026/file/3d167db04a90885ad5208fe8b273668b-Paper-Conference.pdf}
}
```"""
)

TRANSFORMER_ARCHITECTURE = "BTLeelaZeroNet"
SE_ARCHITECTURE = "SeLeelaZeroNet"


def is_transformer(root: pathlib.Path) -> bool:
    """Read the architecture from config.json; it decides the citation."""
    config = root / "config.json"
    if not config.is_file():
        raise SystemExit(
            f"error: {config} is missing; an Lc0 card needs it to tell a "
            "transformer from an SE-ResNet network"
        )
    architectures = json.loads(config.read_text(encoding="utf-8")).get(
        "architectures", []
    )
    if architectures == [TRANSFORMER_ARCHITECTURE]:
        return True
    if architectures == [SE_ARCHITECTURE]:
        return False
    raise SystemExit(
        f"error: {config} has architectures {architectures!r}; expected "
        f"[{TRANSFORMER_ARCHITECTURE!r}] or [{SE_ARCHITECTURE!r}]"
    )


def _citation(*, maia1: bool, transformer: bool) -> str:
    if maia1:
        return _MAIA1_CITATION
    return _LC0_TRANSFORMER_CITATION if transformer else _LC0_SE_CITATION


def generated_section(
    repository_id: str, *, maia1: bool, transformer: bool = False
) -> str:
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

{_citation(maia1=maia1, transformer=transformer)}"""


def update(root: pathlib.Path, *, write: bool, repository_id: str, maia1: bool) -> bool:
    current = common.read_readme(root, required=True)
    transformer = False if maia1 else is_transformer(root)
    section = generated_section(repository_id, maia1=maia1, transformer=transformer)
    updated = common.replace_tail(current, GENERATED_HEADING, section)
    return common.write_or_preview(root, current, updated, write=write)
