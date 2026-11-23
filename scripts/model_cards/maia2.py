"""Complete model cards for the Maia 2 blitz and rapid checkpoints."""

from __future__ import annotations

import json
import pathlib

from model_cards import common

CHECKPOINT_URLS = {
    "blitz": "https://drive.google.com/uc?id=1X-Z4J3PX3MQFJoa8gRt3aL8CIH0PWoyt",
    "rapid": "https://drive.google.com/uc?id=1gbC1-c7c0EQOPPAVpGWubezeEW8grVwc",
}
TRAINING_CONFIG_URL = "https://drive.google.com/uc?id=1GQTskYMVMubNwZH2Bi6AmevI15CS6gk0"


def model_card(repository_id: str, time_control: str) -> str:
    return f"""---
license: mit
library_name: chessnets
pipeline_tag: other
tags:
- chess
- maia
- human-behavior
---

# Maia 2 {time_control.title()}

This is the official Maia-2 checkpoint trained on human {time_control} games,
packaged for inference with ChessNets. Maia-2 conditions its move predictions
on both players' rating categories; it is not a calibrated estimate of an
individual player's strength.

## Use with ChessNets

```python
from chessnets.pipelines.maia2 import Maia2Pipeline

pipeline = Maia2Pipeline.from_pretrained("{repository_id}", device="cpu")
predictions = pipeline(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    elo_self=1500,
    elo_oppo=1600,
)
```

The model returns a legal-move probability distribution and a win-probability
head. It uses the current position rather than a multi-position game history.

## Checkpoint artifacts

- `config.json`: canonical ChessNets inference configuration
- `model.safetensors`: canonical inference tensors
- `provenance.json`: source hashes and exact conversion checks
- `original/model.pt`: original full PyTorch training checkpoint
- `training/original_config.yaml`: original upstream training configuration

The safetensors file contains only `model_state_dict`. The original optimizer
state and training counters remain in `original/model.pt`. Conversion removes
the DataParallel `module.` key prefix, then verifies every tensor and all three
model outputs with exact `torch.equal` comparisons.

Original Maia-2 project: https://github.com/CSSLab/maia2

Original {time_control} checkpoint: {CHECKPOINT_URLS[time_control]}

Original training config:
{TRAINING_CONFIG_URL}

## Intended use and limitations

This checkpoint is intended for research into human move prediction and
skill-conditioned chess behavior. Its predictions reflect patterns in the
training population and time-control domain. They are not guarantees of
playing strength, fairness, instructional quality, or suitability for
high-stakes decisions.

## License

The upstream Maia-2 code and published model are distributed under the MIT
License. ChessNets is a separate software package.

## Citation

```bibtex
@inproceedings{{tang2024maia,
  title={{Maia-2: A Unified Model for Human-{{AI}} Alignment in Chess}},
  author={{Zhenwei Tang and Difan Jiao and Reid McIlroy-Young and Jon Kleinberg and Siddhartha Sen and Ashton Anderson}},
  booktitle={{The Thirty-eighth Annual Conference on Neural Information Processing Systems}},
  year={{2024}},
  url={{https://openreview.net/forum?id=XWlkhRn14K}}
}}
```
"""


def time_control(root: pathlib.Path) -> str:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    value = config.get("time_control")
    if value not in CHECKPOINT_URLS:
        raise ValueError("config.json must select Maia 2 blitz or rapid")
    return value


def update(root: pathlib.Path, *, write: bool, repository_id: str) -> bool:
    current = common.read_readme(root, required=False)
    updated = model_card(repository_id, time_control(root))
    return common.write_or_preview(root, current, updated, write=write)
