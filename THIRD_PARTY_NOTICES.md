# Third-party notices

ChessNets is licensed under the Apache License 2.0 (see [`LICENSE`](LICENSE)).
Some files are adapted from other open-source projects and remain under their
original licenses. Each such file starts with an `SPDX-License-Identifier`
header naming its license and origin. The full license texts are in
[`LICENSES/`](LICENSES).

## Included in the `chessnets` package

| Component | Files | Upstream | License |
|---|---|---|---|
| Allie | `src/chessnets/models/allie/` | [ippolito-cmu/allie](https://github.com/ippolito-cmu/allie) | MIT, Copyright (c) 2025 Yiming Zhang ([text](LICENSES/Allie-MIT.txt)) |
| Maia 2 | `src/chessnets/models/maia2/` | [CSSLab/maia2](https://github.com/CSSLab/maia2) | MIT, Copyright (c) 2024 CSSLab ([text](LICENSES/Maia2-MIT.txt)) |
| OpenCLIP | `src/chessnets/models/chess_clip/modeling_chess_clip.py`, `src/chessnets/models/chess_clip/model_configs/` | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | MIT, Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi, Ludwig Schmidt ([text](LICENSES/OpenCLIP-MIT.txt)) |
| ChessCLIP (from ChessGPT) | same files as OpenCLIP | [waterhorse1/ChessGPT](https://github.com/waterhorse1/ChessGPT) | Apache-2.0, Copyright 2022-2023 Xidong Feng, Yicheng Luo, Ziyan Wang ([text](LICENSES/ChessGPT-Apache-2.0.txt)) |

These files were modified to integrate with ChessNets. The ChessCLIP code
combines the MIT-licensed OpenCLIP fork that ships inside the ChessGPT
repository with the ChessGPT authors' Apache-2.0 changes, so both notices apply.

The prompt formats used by the Carlini pipeline follow [Nicholas Carlini's
2023 experiments](https://nicholas.carlini.com/writing/2023/chess-llm.html) with language models playing chess; ChessNets reimplements
them and includes no upstream source code.

## Not included in the `chessnets` package

The optional `chessnets-lc0-protobuf` adapter in
[`plugins/lc0-protobuf`](plugins/lc0-protobuf) is a separate distribution
licensed under GPL-3.0-or-later (see its [`LICENSE`](plugins/lc0-protobuf/LICENSE)).
It contains code generated from the `net.proto` schema in
[lczero-common](https://github.com/LeelaChessZero/lczero-common) (commit
`55e1b38`, Copyright (C) 2018 The LCZero Authors, GPL-3.0-or-later). The core
`chessnets` package does not import it; it is discovered through an entry
point only when installed.

## Model weights

Checkpoint licenses are separate from code licenses. Each published model's
license is stated on its Hugging Face model card.
