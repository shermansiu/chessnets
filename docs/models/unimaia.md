# UniMaia

Chess policies steered with natural-language instructions for human-like
play.

```shell
pip install "chessnets[unimaia]"
```

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("unimaia", device="cuda")
```

- **Names:** `unimaia`, `unimaia-aux`
- **Paper:** [arXiv:2605.27767](https://arxiv.org/abs/2605.27767)

## Citation

```bibtex
@misc{siu2026unimaiasteeringchesspolicies,
  title         = {{UniMaia}: Steering Chess Policies with Language for Human-like Play},
  author        = {Siu, Sherman and Istead, Lesley},
  year          = {2026},
  eprint        = {2605.27767},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2605.27767}
}
```
