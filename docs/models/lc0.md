# Lc0 networks

Leela Chess Zero policy/value networks, both SE-ResNet (`t`-series) and
transformer (`BT`-series) architectures, converted to safetensors. Playing
strength depends on the source network and on the search settings.

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("lc0_BT4-1024x15x32h-swa-6147500", device="cuda")
```

- **Names:** for example `lc0_BT4-1024x15x32h-swa-6147500`,
  `lc0_t82-768x15x24h-swa-7364000`
- **Upstream:** [Leela Chess Zero](https://lczero.org/)
- **Original `.pb.gz` checkpoints:** install `chessnets[lc0-protobuf]`; see
  [Checkpoint format and Lc0 conversion](../checkpoint-format.md)

## Citation and attribution

Lc0 has no paper. Credit the source network named on the model card and the
[Leela Chess Zero project](https://lczero.org/).
