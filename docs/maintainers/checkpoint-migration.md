# Checkpoint repository migration (maintainers)

This guide is for maintainers converting and publishing first-party checkpoint
repositories, including UniMaia experiments. Users who only load models, or
convert their own Lc0 networks, should read the
[checkpoint format guide](../checkpoint-format.md) instead.

The helper scripts assume the checkpoint Git repositories live in a sibling
`board_game_nn_huggingface/checkpoints` directory; set `CHECKPOINT_ROOT` to
use another location.

ChessNets uses safetensors as the normal runtime format. Protobuf remains an
explicit conversion/reproducibility path supplied by a separate adapter; the
choice is a `from_pretrained` argument and is not part of model architecture.

## Published repository layout

An Lc0 or Maia-1 repository should contain:

```text
config.json
model.safetensors
provenance.json
README.md
original/model.pb.gz
training/original_lczero_training_config.yaml
```

`config.json` is the sole root runtime config. It contains
`model_type: "lc0"`, `config_schema_version: 1`, `architectures`, and a fully
materialized `model` mapping. Materializing defaults prevents a future
ChessNets release from silently changing an old architecture.

The original YAML is retained byte-for-byte under `training/`; it is training
provenance, not a second runtime config. The original protobuf is retained
under `original/` so the conversion can be reproduced.

## Convert Lc0 and Maia-1

To publish a new Lc0 or Maia-1 network, clone (or create) its Hugging Face Git
repository, place the upstream `model.pb.gz` and `config.yaml` at the root,
and run the prepare helper. Preview is the default:

```shell
scripts/prepare_lc0_checkpoint_repo.sh maia-1100
scripts/prepare_lc0_checkpoint_repo.sh --apply maia-1100
```

It accepts several repositories at once. It resolves names under
`../board_game_nn_huggingface/checkpoints`, or accepts an explicit repository
path. The helper refuses dirty or partially prepared repositories, verifies the safetensors Git LFS attribute, preflights protobuf
parsing, previews an idempotent README tail beginning at `## Use with
ChessNets`, and uses `git mv` for the original protobuf and YAML. On `--apply`,
it also writes that README content while preserving the existing frontmatter
and source notes. It never commits or pushes. Set `CHECKPOINT_ROOT` or
`PYTHON_BIN` to override its defaults.

## Model cards

`scripts/update_model_card.py` generates the cards for every first-party
family. Preview is the default, and it accepts several repositories, so an
edited template can be rolled out to all of them at once:

```shell
python scripts/update_model_card.py /path/to/model-repo
python scripts/update_model_card.py --write ../board_game_nn_huggingface/checkpoints/*
```

The family is detected per repository: `maia2` if `config.json` has
`model_type: "maia2"` or the directory name starts with `maia2`, `maia1` for
`maia-*` names, and `lc0` otherwise. Pass `--family lc0|maia1|maia2` to
override it. The repository ID in the usage snippet comes from the Git remote;
pass `--repository-id owner/name` to set it explicitly.

- Lc0 and Maia-1 cards keep the hand-written frontmatter and source notes, and
  regenerate everything from `## Use with ChessNets` onward. Maia-1 cards cite
  the original Maia paper; Lc0 cards credit the source network and the Leela
  Chess Zero project without a made-up model-specific citation.
- Maia 2 cards are generated in full from `config.json`.

The templates live in `scripts/model_cards/` (`lc0.py`, `maia2.py`), with the
shared Git-remote, diff, and write helpers in `common.py`. To add a family, add
a module with an `update(root, *, write, repository_id)` function and register
it in `update_model_card.py`.

From a source checkout, make the legacy adapter importable in the conversion
environment, then preview or run a conversion:

```shell
PYTHONPATH=plugins/lc0-protobuf/src chessnets convert lc0 \
  /path/to/checkpoint-repo /path/to/output --dry-run

PYTHONPATH=plugins/lc0-protobuf/src chessnets convert lc0 \
  /path/to/checkpoint-repo /path/to/output
```

For a directory containing only `model.pb.gz`, pass the architecture config:

```shell
PYTHONPATH=plugins/lc0-protobuf/src chessnets convert lc0 \
  /path/to/model.pb.gz /path/to/output --config /path/to/config.yaml
```

The converter inspects the protobuf before labeling its encoding. The current
adapter accepts `LINEAR16`, decodes learned tensors into float32, saves those
decoded float32 values, and records source/config hashes. `LINEAR16` describes
the encoded protobuf weight layers; it does not describe protobuf metadata or
the safetensors container. Integer PyTorch bookkeeping buffers remain integer.

Conversion retains the protobuf and YAML by default. Use
`--no-retain-protobuf` only for disposable validation output. The four
protobuf-only local directories can therefore remain untouched until an
appropriate config is supplied later.

To force a runtime format during validation:

```python
from chessnets.models.lc0 import AutoLc0Model

legacy = AutoLc0Model.from_pretrained(path, checkpoint_format="protobuf")
portable = AutoLc0Model.from_pretrained(converted_path, use_safetensors=True)
```

Compare state-dict keys and every tensor exactly, then compare policy/value
outputs on representative positions. Automatic loading prefers a sharded
index, then `model.safetensors`, and only then asks for the protobuf adapter.

## Consolidate UniMaia configs

UniMaia and UniMaia-Aux already have safetensors checkpoints. Their migration
only consolidates the trimmed inference YAML and Lc0 YAML into one root JSON:

```shell
python -m chessnets.loaders.convert_unimaia_config /path/to/unimaia-repo
python -m chessnets.loaders.convert_unimaia_config /path/to/unimaia-repo \
  --write --archive-yaml
```

The first command is a non-writing preview. The second writes `config.json`,
embeds the fully materialized Lc0 architecture, moves the old trimmed config to
`legacy/inference_config.yaml`, and moves the Lc0 training YAML to
`training/original_lczero_training_config.yaml`. It does not rewrite the
existing safetensors checkpoint, tokenizer, text config, manifest, or
provenance files.

## Split and convert Maia 2 (archived)

This was a one-off migration; its helper is kept in `scripts/archive/` for
reference and reproducibility. The original `shermansiu/maia2` repository contains blitz and rapid PyTorch
training checkpoints together. Publish them as two self-contained repositories:

```text
shermansiu/maia2-blitz
shermansiu/maia2-rapid
```

Each repository contains:

```text
config.json
model.safetensors
provenance.json
README.md
original/model.pt
training/original_config.yaml
```

`model.safetensors` is the canonical inference artifact. `original/model.pt`
retains the upstream optimizer state, accumulated sample/game counters, and
DataParallel-prefixed state dict for provenance and possible training recovery.
The original YAML is archived byte-for-byte; only the inference architecture
and repository time control are materialized in `config.json`.

Preview or prepare both repositories in a new staging directory:

```shell
PYTHON_BIN=python scripts/archive/migrate_maia2_checkpoint_repo.sh \
  ../board_game_nn_huggingface/checkpoints/maia2 /tmp/maia2-migration

PYTHON_BIN=python scripts/archive/migrate_maia2_checkpoint_repo.sh --apply \
  ../board_game_nn_huggingface/checkpoints/maia2 /tmp/maia2-migration
```

The helper never changes Hugging Face state. It extracts `model_state_dict`,
removes one leading `module.` prefix from each key, writes safetensors, reloads
it, and requires exact `torch.equal` equality for every tensor. It then loads
the original and converted tensors into separate eval-mode models and requires
exact equality for the policy, auxiliary, and value outputs on fixed inputs.
Hashes and the successful checks are recorded in `provenance.json`.

ChessNets accepts the canonical repository IDs, hyphenated short names, and
the older underscore aliases:

```python
from chessnets.pipelines.maia2 import Maia2Pipeline

pipeline = Maia2Pipeline.from_pretrained("shermansiu/maia2-blitz")
legacy_alias = Maia2Pipeline.from_pretrained("maia2_blitz")
```

Loading prefers `model.safetensors` and falls back to a legacy root `model.pt`
when necessary; `use_safetensors=True` requires the safe format and
`use_safetensors=False` explicitly selects the legacy format. The published
layout keeps the full training checkpoint under `original/` instead.

## Licensing boundary

The Apache runtime does not import protobuf or generated protobuf code when it
loads safetensors. The legacy parser, generated schema, and weight decoder live
under `plugins/lc0-protobuf`, a separate distribution with its own publication
gate. It is licensed under GPL-3.0-or-later and keeps its `Private :: Do Not
Upload` classifier until it is ready for PyPI; see `THIRD_PARTY_NOTICES.md`.

The core build sets `source-exclude = ["plugins/**"]`; uv applies source
exclusions to sdists and wheels. The plugin also carries the standard
`Private :: Do Not Upload` classifier as an additional accidental-publication
guard.

`scripts/publish_pypi.sh` runs `scripts/check_distribution_licenses.py` on
every wheel and sdist it builds; it can also be run on artifacts directly. Each
artifact is checked against its own `License-Expression`: every SPDX header
must fit it, and an artifact without a GPL-family license must not contain GNU
license text, protobuf or `*_pb2` imports, or the plugin package. Passing it is
evidence about file contents, not a legal conclusion.

## Model cards and citations

Each model repository should document ChessNets usage, the exact upstream
network/config URLs, source hashes, conversion format and dtype, intended use,
limitations, artifact license, and citations. Maia-1 repositories should cite
McIlroy-Young et al., *Aligning Superhuman AI with Human Behavior: Chess as a
Model System* (KDD 2020), arXiv:2006.01855. Lc0 community networks should
credit the exact network and Lc0 contributors without inventing a paper
citation. Cite ChessNets itself using the repository's `CITATION.cff`.
Maia-2 repositories should cite Tang et al., *Maia-2: A Unified Model for
Human-AI Alignment in Chess* (NeurIPS 2024), OpenReview `XWlkhRn14K`.
