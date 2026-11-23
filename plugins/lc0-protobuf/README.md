# chessnets-lc0-protobuf

Optional adapter for loading and converting legacy Lc0 and Maia-1
`model.pb.gz` checkpoints with ChessNets.

Most users do not need it: published ChessNets checkpoints use safetensors,
which the core `chessnets` package loads directly. Install the adapter only to
convert a protobuf network yourself, for example with `chessnets convert lc0`.

## License

Unlike the Apache-2.0 core package, this adapter is licensed under
GPL-3.0-or-later (see
[`LICENSE`](https://github.com/shermansiu/chessnets/blob/main/plugins/lc0-protobuf/LICENSE)),
because its `net_pb2.py` is generated from the GPL-3.0-or-later
[lczero-common](https://github.com/LeelaChessZero/lczero-common) `net.proto`
schema. It is distributed separately and discovered by ChessNets through the
`chessnets.lc0_checkpoint_loaders` entry point, so installing `chessnets` alone
does not include it.

## Installation

```shell
pip install "chessnets[lc0-protobuf]"
```

Once installed, no import or extra argument is needed: loading a repository
or path that has a `model.pb.gz` but no safetensors checkpoint uses the
adapter automatically, and safetensors is still preferred when both exist.

From a ChessNets source checkout, install it with:

```shell
pip install ./plugins/lc0-protobuf
```
