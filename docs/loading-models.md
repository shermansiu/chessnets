# Loading models

`AutoPipeline.from_pretrained()` builds the right pipeline for a model from
its name, a Hugging Face repository ID, or a local directory:

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("maia-1500", device="cpu")
```

## Built-in names

Published checkpoints can be loaded by their full repository ID or by a short
name:

| Family | Examples |
|---|---|
| Maia-1 | `maia-1100` … `maia-1900`, `maia-2200` |
| Maia 2 | `maia2-blitz`, `maia2-rapid` |
| Lc0 | `lc0_BT4-1024x15x32h-swa-6147500`, `lc0_t82-768x15x24h-swa-7364000`, … |
| UniMaia | `unimaia`, `unimaia-aux` |
| Allie | `allie_medium_best` |
| ChessGPT | `Waterhorse/chessgpt-base-v1`, `Waterhorse/chessgpt-chat-v1_simple`, … |
| ChessCLIP | `Waterhorse/ChessCLIP` |

The full list is available at runtime:

```python
from chessnets.pipelines import PIPELINE_INFOS

print(sorted(PIPELINE_INFOS))
```

Some families need an optional extra, such as `chessnets[maia2]` or
`chessnets[text-generation]`. A missing dependency raises an `ImportError`
that names the extra to install.

## Any repository with a `config.json`

A repository that is not registered by name still loads if its root
`config.json` names a supported model architecture:

```json
{
  "architectures": ["UniMaiaModel"]
}
```

Supported architectures are `BTLeelaZeroNet`, `SeLeelaZeroNet`, `Maia2Model`,
`UniMaiaModel`, and `CausalLMWithRegressionHead` (Allie).

ChessNets resolves a model as follows:

1. A registered name or repository ID always wins, and needs no network
   access to resolve.
2. Otherwise, ChessNets reads `config.json` from the local directory, or from
   the Hugging Face repository at the same revision used for the weights.
3. The first entry in `architectures` with a registered factory is used.
4. If no entry matches, the error lists the architectures found and how to
   register one.

Architecture names are plain identifiers. ChessNets never imports or runs code
named in a repository's metadata. Hub options such as `revision`, `token`,
`cache_dir`, and `local_files_only` apply both to reading `config.json` and to
loading the weights.

## Registering your own models

Register a name when a checkpoint needs defaults that its architecture alone
does not imply:

```python
from chessnets import AutoPipeline
from chessnets.pipelines.lc0 import Lc0Pipeline


def create_my_net(device, **kwargs):
    return Lc0Pipeline.from_pretrained("me/my-lc0-net", device=device, **kwargs)


AutoPipeline.register("my-net", create_my_net, aliases=["me/my-lc0-net"])
```

Register an architecture to make every repository whose `config.json` names it
loadable:

```python
def create_from_architecture(model_id_or_path, device, **kwargs):
    ...


AutoPipeline.register_architecture("MyModel", create_from_architecture)
```

Names are matched exactly and case-sensitively. Registering a name that is
already taken raises `ValueError` unless you pass `overwrite=True`.
