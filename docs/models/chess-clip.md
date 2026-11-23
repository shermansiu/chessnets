# ChessCLIP

A CLIP-style model from the ChessGPT project that embeds board positions and
chess text in a shared space. ChessNets scores each legal move against a
text prompt, so `make_prediction` takes `prompts` as well as PGNs.

```shell
pip install "chessnets[chess-clip]"
```

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained("Waterhorse/ChessCLIP", device="cuda")
```

- **Names:** `Waterhorse/ChessCLIP`
- **Upstream:** [waterhorse1/ChessGPT](https://github.com/waterhorse1/ChessGPT)
- **Paper:** [arXiv:2306.09200](https://arxiv.org/abs/2306.09200) (shared with ChessGPT)

## Citation

```bibtex
@article{feng2023chessgpt,
  title   = {{ChessGPT}: Bridging Policy Learning and Language Modeling},
  author  = {Feng, Xidong and Luo, Yicheng and Wang, Ziyan and Tang, Hongrui and Yang, Mengyue and Shao, Kun and Mguni, David and Du, Yali and Wang, Jun},
  journal = {arXiv preprint arXiv:2306.09200},
  year    = {2023}
}
```
