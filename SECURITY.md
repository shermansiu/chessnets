# Security policy

## Reporting a vulnerability

Please report security problems privately, through
[GitHub's private vulnerability reporting](https://github.com/shermansiu/chessnets/security/advisories/new)
(**Security → Report a vulnerability** on the repository), not in a public
issue. Include what is affected, how to reproduce it, and the ChessNets,
PyTorch and Python versions. You should get a reply within a week.

Fixes are made on the latest release only.

## What ChessNets trusts

Some of what ChessNets does reads files or runs code you choose, by design:

- **Checkpoints.** Lc0, Maia-1, Maia 2, UniMaia and Allie weights load from
  safetensors or with `torch.load(..., weights_only=True)`, which cannot run
  code. ChessCLIP's published checkpoint is a full pickle, so it is loaded
  only from its authors' repository, `Waterhorse/ChessCLIP`. Load other
  checkpoints only from sources you trust.
- **Transformers models** (ChessGPT, UniMaia's language model) load through
  Hugging Face Transformers. ChessNets never passes `trust_remote_code=True`.
- **Plugins.** Installed packages that register a
  `chessnets.lc0_checkpoint_loaders` entry point, such as
  `chessnets-lc0-protobuf`, are ordinary Python code and run when a checkpoint
  needs them.
- **Configuration files** for the UCI engine and Lichess bot are read with
  `yaml.safe_load`.
- **API tokens.** The Lichess bot sends its token only to the configured
  `base_url` (`https://lichess.org` by default), and the OpenAI pipelines use
  the OpenAI client's own configuration. Neither is written to disk or logs.

A problem that goes beyond what is described here, such as a checkpoint or
configuration file being able to run code, is a vulnerability worth reporting.
