# Maia 2

A single human move-prediction model conditioned on both players' ratings,
with separate blitz and rapid checkpoints.

```shell
pip install "chessnets[maia2]"
```

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("maia2-rapid", device="cpu")
```

- **Names:** `maia2-blitz`, `maia2-rapid`
- **Upstream:** [CSSLab/maia2](https://github.com/CSSLab/maia2)
- **Paper:** [OpenReview](https://openreview.net/forum?id=XWlkhRn14K)

## Citation

```bibtex
@inproceedings{tang2024maia,
  title     = {Maia-2: A Unified Model for Human-{AI} Alignment in Chess},
  author    = {Tang, Zhenwei and Jiao, Difan and McIlroy-Young, Reid and Kleinberg, Jon and Sen, Siddhartha and Anderson, Ashton},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year      = {2024},
  url       = {https://openreview.net/forum?id=XWlkhRn14K}
}
```
