# Contributing models and pipelines

This guide describes the repository conventions for adding a model family, a
pipeline, or a published checkpoint. A checkpoint is a set of weights and
configuration; a model turns tensors into tensors; a pipeline owns chess input
preparation, batching, inference, and user-facing predictions. Keep those
responsibilities separate even when a model has only one pipeline.

## Choose the smallest change

- For another checkpoint of an existing architecture, prefer metadata and an
  `AutoPipeline` registration. Do not copy a model or pipeline implementation.
- For another architecture with the same input and output contract, add the
  model implementation and reuse the pipeline.
- Add a new pipeline when preprocessing, required inputs, output semantics, or
  the inference backend changes.

## Repository layout

A built-in family normally has this shape:

```text
src/chessnets/
  models/<family>/
    __init__.py
    configuration_<family>.py
    modeling_<family>.py
    layers_<family>.py          # only when useful
  pipelines/<family>/
    __init__.py
    pipeline_<family>.py
  loaders/load_<family>_model.py  # only for complex/legacy formats
tests/
  test_<family>_config.py
  test_<family>_model.py
  test_<family>_pipeline.py
```

Use lower-case snake case for modules and `FamilyConfig`, `FamilyModel`, and
`FamilyPipeline` for public classes. A family with several architectures puts
each in `modeling_<family>_<architecture>.py` and names its class
`Family<Architecture>Model`, as `models/lc0/` does with `Lc0ChessFormerModel`
and `Lc0SeResNetModel`. Put variants that differ only in prompts or inputs
behind a frozen config dataclass (such as `TextGenerationPipelineConfig`)
rather than in subclasses. Put shared checkpoint resolution in
`models/_pretrained.py` and shared inference mechanics in
`pipelines/pipeline_utils.py`. Avoid adding family-specific behavior to those
modules.

## Model contract

A model contribution should provide the following where they apply:

- A configuration object that can be constructed from a dictionary and loaded
  from a local directory or Hugging Face repository. Existing native chess
  models use `config.yaml`; new metadata-based auto-dispatch will additionally
  use the repository-root `config.json` described below.
- `Model.from_config(config, **kwargs)` to construct an uninitialized model.
- `Model.from_pretrained(pretrained_model_name_or_path, **kwargs)` accepting a
  local directory or Hugging Face repository ID. Use
  `resolve_pretrained_file()` so `revision`, `cache_dir`, `token`, and
  `local_files_only` have consistent meanings.
- Explicit checkpoint validation: required files, supported serialization
  formats, missing/unexpected state-dict keys, incompatible dimensions, and
  useful error messages containing the model ID or path.
- Device and dtype handling with no unconditional CUDA use. Load onto CPU when
  that avoids an unnecessary memory peak, then move once.
- Inference-safe behavior. The pipeline calls `eval()`, but a directly loaded
  model should not silently enable gradients or mutate global Torch settings.

Do not execute Python from a model repository or implicitly enable a
`trust_remote_code` equivalent. Do not use `pickle`-based weights from an
untrusted source without making the risk explicit. Prefer `safetensors` for new
published weights.

If an architecture-dispatching model class is needed, follow `AutoLc0Model`:
load and validate configuration first, dispatch through an explicit mapping,
and let the selected class load the weights. Never infer an architecture from
tensor shapes when stable metadata exists.

## Pipeline contract

Subclass `pipelines.base.Pipeline` and implement:

- `default_input_config()`, declaring all required, optional, paired, and
  broadcastable inputs with canonical input names;
- `from_pretrained(pretrained_model_name_or_path, device="cpu", **kwargs)`,
  loading the backend and returning `cls(...)` rather than the concrete base
  class;
- `forward(...)` for tensor/backend inference;
- `make_prediction(...)` for normalization, batching, preprocessing, and
  stable output objects from `pipelines.outputs`.

Constructors should continue to accept an already-created model. This keeps
unit tests cheap and supports dependency injection. Separate loader arguments
(`revision`, `token`, `dtype`, and similar) from pipeline options
(`temperature`, input mode, and batching behavior); reject conflicts rather
than silently choosing one. Use `torch.inference_mode`, call `model.eval()`,
and honor the requested device.

For datasets, support one item and a batch, validate equal lengths before
inference, keep `collate_fn` deterministic, and use the adaptive helpers in
`pipeline_utils.py` instead of creating process pools for small requests.
Return a clear error for an empty primary input unless the established pipeline
contract explicitly permits it.

Optional backends must stay optional. Import them as late as practical, add the
dependency to the narrowest extra in `pyproject.toml`, and translate a missing
optional package into an error that names the install extra. Importing
`AutoPipeline` must not import Transformers, OpenAI, Outlines, or OpenCLIP.

Export public classes from the family `__init__.py`. Use `Pipeline` terminology
throughout; the former `Predictor` API has been removed. Pipeline inputs use
plural names such as `pgns`, `fens`, and `prompts` because every pipeline API
accepts either one item or a batch.

## Registering a pipeline

`AutoPipeline` currently resolves exact model IDs and aliases from
`pipelines/auto_pipeline.py`. Register a factory rather than a preloaded model.
The factory must accept `device` plus keyword options and defer heavyweight
imports and downloads until called. Use aliases sparingly: they are global,
case-sensitive names and must not depend on network state.

Curated registrations remain appropriate when a checkpoint needs defaults
that are not intrinsic to its architecture, such as the UniMaia input mode.
External users can call `AutoPipeline.register(...)`; built-ins should be added
in `_register_builtins()` with a collision test.

## Add auto-loading metadata

For a model that should load through `AutoPipeline` without a hard-coded
repository ID, publish a repository-root `config.json` containing its model
architecture:

```json
{
  "architectures": ["UniMaiaModel"]
}
```

The value must be the public model class name, not a pipeline class, module
path, or Python expression. Register that exact architecture name with its
pipeline factory in the architecture registry, and add a test that loads the
model from a temporary local repository by metadata alone. Add explicit
architecture aliases only for class names already published in checkpoint
metadata; matching is case-sensitive and is not fuzzy.

Keep an exact-ID registration as well when a checkpoint needs curated defaults
that cannot be derived from its architecture. Exact-ID registrations take
precedence over metadata.

How names and `config.json` metadata are resolved is described in
[Loading models](docs/loading-models.md).

## Contribution checklist

Before opening a change, verify:

- [ ] The model, config, and pipeline responsibilities are separated.
- [ ] Local-directory and Hugging Face loading work through `from_pretrained`.
- [ ] Hub options and device/dtype behavior are documented and tested.
- [ ] Checkpoint files and configuration are validated with actionable errors.
- [ ] Input contracts cover scalar, batch, empty, mismatched, and game-over
      cases relevant to the pipeline.
- [ ] Public exports, `AutoPipeline` registrations, optional extras, and
      compatibility aliases are updated where applicable.
- [ ] Import-laziness tests still pass without optional packages installed.
- [ ] Tests avoid downloading large checkpoints; network integration tests are
      explicitly gated.
- [ ] `python -m pytest` passes and changed Python files are formatted with
      Ruff (or `pre-commit run --all-files`).
- [ ] A new model family has a page in [`docs/models/`](docs/models/) with
      its shortest loading call, upstream links and citation, and a row in
      the README's supported-models table linking to it.

## Licensing of contributed code

ChessNets is Apache-2.0. Code adapted from another project keeps its original
license: add an `SPDX-License-Identifier` header naming the license and
upstream project, put the license text in `LICENSES/`, and add a row to
`THIRD_PARTY_NOTICES.md`. Only permissive, Apache-compatible licenses can go
in the core package. Copyleft code (GPL, AGPL) belongs in a separately
distributed plugin such as `plugins/lc0-protobuf`.

## Maintainer guides

Guides for converting and publishing first-party checkpoints are in
[`docs/maintainers/`](docs/maintainers/).

