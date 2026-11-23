# Checkpoint format

ChessNets checkpoints are Hugging Face-style repositories that store weights
as [safetensors](https://huggingface.co/docs/safetensors). The core package
loads them without any legacy parser. Lc0's original protobuf format is
supported only through a separate, optional adapter.

## Repository layout

An Lc0 or Maia-1 repository contains:

```text
config.json
model.safetensors
provenance.json
README.md
original/model.pb.gz
training/original_lczero_training_config.yaml
```

A Maia 2 repository has the same shape, with `original/model.pt` and
`training/original_config.yaml` holding the upstream PyTorch checkpoint and
training configuration.

- **`config.json`** is the only runtime configuration. For Lc0 networks it has
  `model_type: "lc0"`, `config_schema_version: 1`, `architectures`, and a
  `model` mapping with every architecture default written out, so a future
  ChessNets release cannot silently change how an old network is built.
- **`model.safetensors`** holds float32 weights.
- **`provenance.json`** records source URLs, hashes, and the checks performed
  during conversion.
- **`original/`** and **`training/`** keep the upstream files byte-for-byte so
  a conversion can be reproduced. They are not read at runtime.

The `architectures` field also lets [`AutoPipeline`](loading-models.md) load a
repository without a registered name.

## How loading chooses a format

Loading prefers a sharded safetensors index, then `model.safetensors`, and
only then falls back to the protobuf adapter. You can request a format
explicitly:

```python
from chessnets.models.lc0 import AutoLc0Model

model = AutoLc0Model.from_pretrained(path, use_safetensors=True)
legacy = AutoLc0Model.from_pretrained(path, checkpoint_format="protobuf")
```

## Converting an Lc0 network

To use an Lc0 network that is only available as `model.pb.gz`, convert it once
with the protobuf adapter. The adapter is GPL-3.0-or-later and distributed
separately from the Apache-2.0 core (see
[its README](../plugins/lc0-protobuf/README.md)):

```shell
pip install "chessnets[lc0-protobuf]"
# or, from a ChessNets source checkout: pip install ./plugins/lc0-protobuf

chessnets convert lc0 /path/to/checkpoint-repo /path/to/output --dry-run
chessnets convert lc0 /path/to/checkpoint-repo /path/to/output
```

For a bare `model.pb.gz` without a `config.json`, pass the architecture
configuration:

```shell
chessnets convert lc0 /path/to/model.pb.gz /path/to/output \
  --config /path/to/config.yaml
```

The converter reads the protobuf's weight encoding (currently `LINEAR16`),
decodes the learned tensors to float32, and records source and config hashes
in `provenance.json`. It copies the protobuf to `original/model.pb.gz` unless
you pass `--no-retain-protobuf`, which is only advisable for throwaway output.

## Verifying a conversion

Load the source and the converted repository side by side, then compare
state-dict keys and every tensor exactly before comparing policy and value
outputs on representative positions:

```python
import torch

from chessnets.models.lc0 import AutoLc0Model

legacy = AutoLc0Model.from_pretrained(source_path, checkpoint_format="protobuf")
converted = AutoLc0Model.from_pretrained(output_path, use_safetensors=True)

legacy_state, converted_state = legacy.state_dict(), converted.state_dict()
assert legacy_state.keys() == converted_state.keys()
assert all(torch.equal(legacy_state[k], converted_state[k]) for k in legacy_state)
```

## Model cards and citations

Each published repository's `README.md` documents ChessNets usage, the
upstream network and configuration, intended use, limitations, the weights'
license, and citations. Weights keep their own licenses, separate from the
ChessNets code license.

- **Maia-1:** McIlroy-Young et al., *Aligning Superhuman AI with Human
  Behavior: Chess as a Model System*, KDD 2020
  ([arXiv:2006.01855](https://arxiv.org/abs/2006.01855)).
- **Maia 2:** Tang et al., *Maia-2: A Unified Model for Human-AI Alignment in
  Chess*, NeurIPS 2024 ([OpenReview](https://openreview.net/forum?id=XWlkhRn14K)).
- **Lc0 networks:** credit the specific network and the Leela Chess Zero
  contributors.
