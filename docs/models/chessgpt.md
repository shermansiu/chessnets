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

Multiple-choice mode generates one position at a time, because Outlines
applies one constraint to a whole batch and each position has different legal
moves. `batch_size` then only sets how many prompts are prepared together,
and a value above 1 warns once; `batch_size="auto"` uses 1.

`experimental_batched_choice=True` generates a whole batch at once instead,
with each position restricted to its own legal moves. With ChessGPT-base on an
RTX 5090 it was 3.6× faster at `batch_size=16` and 3.9× at 32. On 256
positions it picked the same move as unbatched generation in 249 to 250,
depending on the batch size. Where it differed, the two moves' scores were
equal or one bfloat16 rounding step apart, and padding tipped them the other
way. It is experimental: results can differ slightly from unbatched runs and
between batch sizes.

- **Names:** `Waterhorse/chessgpt-base-v1`, `Waterhorse/chessgpt-chat-v1_simple`
- **Upstream:** [waterhorse1/ChessGPT](https://github.com/waterhorse1/ChessGPT)
- **Paper:** [arXiv:2306.09200](https://arxiv.org/abs/2306.09200)

## Profiles

Each registered name is a ChessGPT checkpoint with a prompt format:

| Name | Checkpoint | Prompt format | Inputs |
|---|---|---|---|
| `Waterhorse/chessgpt-chat-v1` | chat | JSON move probabilities | PGN, prompt |
| `Waterhorse/chessgpt-chat-v1_simple` | chat | `chat` | PGN, prompt |
| `Waterhorse/chessgpt-chat-v1_simple_original` | chat | `chat_original` | PGN, prompt |
| `Waterhorse/chessgpt-chat-v1_general_policy` | chat | `general_policy` | PGN, time control, Elos |
| `Waterhorse/chessgpt-base-v1` | base | `pgn_header` | PGN, prompt |
| `Waterhorse/chessgpt-base-v1_with-time-control-and-elo` | base | `pgn_header_with_time_control_and_elo` | PGN, time control, Elos |

The same formats work with other causal language models through
`TextGenerationPipeline`:

```python
from chessnets.pipelines.text_generation import (
    TextGenerationPipeline,
    TextGenerationPipelineConfig,
)

pipeline = TextGenerationPipeline.from_pretrained(
    "Waterhorse/chessgpt-base-v1",
    device="cuda",
    pipeline_config=TextGenerationPipelineConfig(prompt_format="pgn_header"),
)
predictions = pipeline.make_prediction("1. e4 e5 2. Nf3", prompt="")
```

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
