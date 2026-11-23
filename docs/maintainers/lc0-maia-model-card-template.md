---
license: REPLACE_WITH_VERIFIED_ARTIFACT_LICENSE
library_name: chessnets
pipeline_tag: board-game
---

# MODEL_NAME

Briefly identify the Lc0 network or Maia rating target and architecture.

## Usage

```python
from chessnets.pipelines.lc0 import Lc0Pipeline

pipeline = Lc0Pipeline.from_pretrained("REPOSITORY_ID")
prediction = pipeline("1. e4 e5")
```

## Artifacts and conversion

- Runtime config: `config.json` (schema version 1)
- Runtime weights: `model.safetensors` (decoded float32 tensors)
- Original weights: `original/model.pb.gz`
- Original training config: `training/original_lczero_training_config.yaml`
- Source URLs and revisions: REPLACE
- Source SHA-256: REPLACE
- Config SHA-256: REPLACE
- Conversion tool/version: REPLACE

The protobuf weight layers used `LINEAR16`; conversion decoded those values to
float32. Safetensors does not restore precision already lost in the source
encoding.

## Intended use and limitations

Describe playing strength or human-move target, supported inputs, known biases,
and why policy probabilities should not be interpreted as calibrated advice.

## License and attribution

State the verified model-artifact license separately from the ChessNets code
license. Retain all required upstream copyright and attribution notices.

## Citation

For Maia-1, include the official citation for McIlroy-Young et al. (KDD 2020),
arXiv:2006.01855. For an Lc0 community network, credit its exact network page
and contributors. Add the ChessNets software citation when available.
