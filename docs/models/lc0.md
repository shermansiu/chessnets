# Lc0 networks

Leela Chess Zero policy/value networks, converted to safetensors. Lc0 names
its training runs `T` ("test") and `BT` ("big transformer") followed by a
number, and the architecture depends on the run, not the prefix: for example,
[T60 to T79](https://lczero.org/play/networks/legacy/) are SE-ResNets. Every
network registered here (from runs T3, T82, BT3 and BT4) is a transformer
using the Chessformer architecture. ChessNets also supports the SE-ResNet
architecture, which Maia-1 uses. Playing strength depends on the source
network and on the search settings.

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("lc0_BT4-1024x15x32h-swa-6147500", device="cuda")
```

- **Names:** `lc0_t3-512x15x16h-swa-2815000`, `lc0_t82-768x15x24h-swa-7364000`,
  `lc0_BT3-768x15x24h-swa-2790000`, `lc0_BT4-1024x15x32h-swa-5000000`,
  `lc0_BT4-1024x15x32h-swa-6147500`, `lc0_BT4-spsa-1740`
- **Upstream:** [Leela Chess Zero](https://lczero.org/)
- **Papers (Chessformer):** [arXiv:2409.12272](https://arxiv.org/abs/2409.12272)
  (2024) and [arXiv:2605.19091](https://arxiv.org/abs/2605.19091) (ICLR 2026)
- **Original `.pb.gz` checkpoints:** install `chessnets[lc0-protobuf]`; see
  [Checkpoint format and Lc0 conversion](../checkpoint-format.md)

## Citation and attribution

Credit the source network named on the model card and the
[Leela Chess Zero project](https://lczero.org/). For the transformer networks,
also cite the Chessformer papers:

```bibtex
@misc{monroe2024mastering,
  title         = {Mastering Chess with a Transformer Model},
  author        = {Monroe, Daniel and Chalmers, Philip A.},
  year          = {2024},
  eprint        = {2409.12272},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2409.12272}
}

@inproceedings{monroe2026chessformer,
  title     = {{Chessformer}: A Unified Architecture for Chess Modeling},
  author    = {Monroe, Daniel and Eilender, George and Chalmers, Philip and Tang, Zhenwei and Anderson, Ashton},
  booktitle = {International Conference on Learning Representations},
  editor    = {C. Vondrick and B. Hariharan and C. Raffel and L. Pinto and D. Yang and A. Faust},
  pages     = {36245--36271},
  volume    = {2026},
  year      = {2026},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2026/file/3d167db04a90885ad5208fe8b273668b-Paper-Conference.pdf}
}
```

The SE-ResNet networks have no paper of their own; credit the source network
and the Lc0 project.
