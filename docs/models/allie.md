# Allie

A transformer trained on human games that predicts moves, resignations and
clock usage, designed to be paired with a small amount of search.

```shell
pip install "chessnets[allie]"
```

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("allie_medium_best", device="cuda")
```

- **Names:** `allie_medium_best`
- **Upstream:** [ippolito-cmu/allie](https://github.com/ippolito-cmu/allie)
- **Paper:** [arXiv:2410.03893](https://arxiv.org/abs/2410.03893)

## Citation

```bibtex
@inproceedings{zhang2025allie,
  title     = {Human-Aligned Chess With a Bit of Search},
  author    = {Zhang, Yiming and Jacob, Athul Paul and Lai, Vivian and Fried, Daniel and Ippolito, Daphne},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  url       = {https://arxiv.org/abs/2410.03893}
}
```
