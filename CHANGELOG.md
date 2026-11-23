# Changelog

All notable changes to `chessnets` and `chessnets-lc0-protobuf` are recorded
here. The two packages are released together and share one version. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
