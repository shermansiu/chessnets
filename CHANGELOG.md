# Changelog

All notable changes to `chessnets` and `chessnets-lc0-protobuf` are recorded
here. The two packages are released together and share one version. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `batch_size="auto"` when calling a pipeline. It starts from a batch size
  suited to the pipeline and its device, halves it whenever a batch runs out
  of memory, and keeps the size that fits; `last_batch_size` records it.
- `TextGenerationPipeline`, configured by `TextGenerationPipelineConfig`
  (`prompt_format` and `constraint`), for any causal language model.
- `SECURITY.md`, which says how to report vulnerabilities privately and what
  ChessNets trusts when it loads checkpoints and plugins.
- `py.typed`, so type checkers use ChessNets' annotations.

### Changed

- Importing `AutoPipeline` or `PIPELINE_INFOS` no longer imports PyTorch or
  any pipeline, so looking up model names takes about 0.06 s instead of
  about 1.7 s. Each pipeline is imported when it is first created.
- **Breaking:** public names now follow the `FamilyConfig`, `FamilyModel` and
  `FamilyPipeline` convention. The registered model names are unchanged.
  - The Lc0 modules are one package, `chessnets.models.lc0`, holding
    `configuration_lc0`, `modeling_lc0`, `modeling_lc0_chessformer`,
    `modeling_lc0_seresnet`, their `layers_*` modules and `policy_map`.
    `BTLeelaZeroNet` is `Lc0ChessFormerModel`, `SeLeelaZeroNet` is
    `Lc0SeResNetModel`, and every `Leela*` configuration class is `Lc0*`
    (`LeelaNetConfig` is `Lc0Config`, `LeelaInferenceConfig` is
    `Lc0InferenceConfig`). Checkpoints whose `config.json` names the old
    classes still load.
  - Maia 2's `Config` is `Maia2Config`.
  - ChessCLIP's `ChessCLIP`, `ChessClipPipeline`, `ChessClipDataset` and
    `OpenClipTokenizer` are `ChessCLIPModel`, `ChessCLIPPipeline`,
    `ChessCLIPDataset` and `OpenCLIPTokenizer`.
  - The six `HFTransformersSimpleTextPipeline*` classes are
    `TextGenerationPipeline` with a `TextPromptFormat`: `chat`,
    `chat_original`, `general_policy`, `pgn_header` (also replacing
    `LocalHFTransformersSimpleTextPipelinePgnHeader`; pass
    `attn_implementation=None`) and `pgn_header_with_time_control_and_elo`.
    The prompts they send are unchanged.
  - `HFTransformersProbabilityTextPipeline`, `OutlinesProbabilityTextPipeline`
    and `(Async)OpenAIProbabilityTextPipeline` are
    `TransformersProbabilityPipeline`, `OutlinesProbabilityPipeline` and
    `(Async)OpenAIProbabilityPipeline`.
  - The text-generation dataset classes are private.
  - The Carlini pipelines are `CarliniPipeline` and `AsyncCarliniPipeline`,
    with the variant chosen by `CarliniPipelineConfig(input_mode=...)`;
    `num_tokens` is keyword-only, and the dataset classes are private.
- The `text-generation` extra requires Outlines 1.2.10 or later, which applies
  the tokenizer's chat template itself, and no longer installs `jinja2` or
  `packaging`. The fallback for older Outlines, which rendered the template
  with an extra system message and so sent different prompts, is gone.
- Allie checkpoints load with `weights_only=True` on every PyTorch version.

## [0.1.1] - 2026-11-23

### Added

- `chessnets[all]`, which installs every runtime extra. It leaves out
  `lc0-protobuf`, which is GPL-3.0-or-later and stays opt-in.
- A Zenodo DOI ([10.5281/zenodo.24458166](https://doi.org/10.5281/zenodo.24458166))
  in `CITATION.cff` and the README citation.
- Documentation at [chessnets.readthedocs.io](https://chessnets.readthedocs.io/),
  linked from the PyPI project page along with the changelog and issue tracker.

## [0.1.0] - 2026-11-23

First public release.

### Added

- One PyTorch API for Lc0, Maia-1, Maia 2, UniMaia, Allie, ChessGPT and
  ChessCLIP: `AutoPipeline.from_pretrained` loads checkpoints from the Hugging
  Face Hub, local directories, or any repository whose `config.json` names a
  supported architecture, and `make_prediction` batches PGNs automatically.
- Lc0 SE-ResNet and transformer networks loaded from safetensors in the core
  install, with no protobuf dependency.
- `chessnets-lc0-protobuf` (GPL-3.0-or-later), an optional plugin installed
  with `chessnets[lc0-protobuf]` that reads original Lc0 `.pb.gz` checkpoints.
- A UCI engine (`chessnets uci`, also installed as `chessnets-uci`) and a
  Lichess bot runner (`chessnets lichess-bot`) for any supported model.
- PGNs that start from a `[FEN]` header, and a clear `ValueError` for
  positions with no legal moves; drawn positions emit `DrawnPositionWarning`.

[Unreleased]: https://github.com/shermansiu/chessnets/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/shermansiu/chessnets/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/shermansiu/chessnets/releases/tag/v0.1.0
