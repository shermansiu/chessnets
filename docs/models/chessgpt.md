# ChessGPT

Language models trained on chess games and chess text. ChessNets turns their
text output into moves. By default a regular expression keeps output in move
notation; `constraint="choice"` restricts it to the legal moves in each
position.

```shell
pip install "chessnets[text-generation]"
```

```python
from chessnets import AutoPipeline

pipeline = AutoPipeline.from_pretrained(
    "Waterhorse/chessgpt-base-v1", device="cuda", constraint="choice"
)
```

- **Names:** `Waterhorse/chessgpt-base-v1`, `Waterhorse/chessgpt-chat-v1_simple`
- **Upstream:** [waterhorse1/ChessGPT](https://github.com/waterhorse1/ChessGPT)
- **Paper:** [arXiv:2306.09200](https://arxiv.org/abs/2306.09200)

The same extra also provides Carlini-style PGN prompting and
OpenAI-compatible endpoints. For the prompting format, credit Nicholas
Carlini's [2023 experiments](https://nicholas.carlini.com/writing/2023/chess-llm.html).

## Citation

```bibtex
@article{feng2023chessgpt,
  title   = {{ChessGPT}: Bridging Policy Learning and Language Modeling},
  author  = {Feng, Xidong and Luo, Yicheng and Wang, Ziyan and Tang, Hongrui and Yang, Mengyue and Shao, Kun and Mguni, David and Du, Yali and Wang, Jun},
  journal = {arXiv preprint arXiv:2306.09200},
  year    = {2023}
}
```
