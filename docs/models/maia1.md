# Maia-1

Human move-prediction networks trained on Lichess games at fixed rating
bands. They use the Lc0 SE-ResNet architecture and run through the Lc0
pipeline, so the core install is enough.

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("maia-1500", device="cpu")
```

- **Names:** `maia-1100` … `maia-1900`, `maia-2200`
- **Upstream:** [CSSLab/maia-chess](https://github.com/CSSLab/maia-chess)
- **Paper:** [arXiv:2006.01855](https://arxiv.org/abs/2006.01855)

## Citation

```bibtex
@inproceedings{mcilroyyoung2020maia,
  title     = {Aligning Superhuman {AI} with Human Behavior: Chess as a Model System},
  author    = {McIlroy-Young, Reid and Sen, Siddhartha and Kleinberg, Jon and Anderson, Ashton},
  booktitle = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year      = {2020},
  doi       = {10.1145/3394486.3403219}
}
```
