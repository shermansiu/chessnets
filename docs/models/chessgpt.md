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

## Faster generation

Two experimental options, both on by default, make generation faster:

- `batched_choice` generates a whole batch at once with `constraint="choice"`,
  each position restricted to its own legal moves. Outlines applies one
  constraint to a whole batch, so ChessNets keeps a separate one for each row.
  It has no effect with `constraint="regex"`.
- `sort_by_length` batches prompts of similar length together, longest first,
  so less of each batch is padding. Predictions still come back in the input
  order. It applies to every local ChessGPT profile, but helps the JSON
  `Waterhorse/chessgpt-chat-v1` profile little (about 1.05×), since generating
  its long answers takes most of the time.

With ChessGPT-base, `constraint="choice"`, on an RTX 5090 and 256 positions:

| Run | Speed vs one at a time | Same move as one at a time |
|---|---|---|
| `batched_choice`, `batch_size=16` | 3.8× | 250/256 |
| `batched_choice`, `batch_size=32` | 4.2× | 249/256 |
| Both options, `batch_size=16` | 4.2× | 251/256 |
| Both options, `batch_size=32` | 4.6× | 249/256 |

Padding changes the model's numbers slightly. Every differing move was a
near-tie: the two moves' scores were equal, or one bfloat16 rounding step
apart, where the answers parted. So results can differ slightly from runs
without these options and between batch sizes. To reproduce results made
before they existed, turn them off:

```python
pipeline = AutoPipeline.from_pretrained(
    "Waterhorse/chessgpt-base-v1",
    device="cuda",
    constraint="choice",
    batched_choice=False,
    sort_by_length=False,
)
```

With `batched_choice=False`, `constraint="choice"` generates one position at a
time. `batch_size` then only sets how many prompts are prepared together, a
value above 1 warns once, and `batch_size="auto"` uses 1.

[`benchmarks/text_generation/benchmark_choice_batching.py`](https://github.com/shermansiu/chessnets/blob/main/benchmarks/text_generation/benchmark_choice_batching.py)
measures this for other models and positions.

## Citation

```bibtex
@article{feng2023chessgpt,
  title   = {{ChessGPT}: Bridging Policy Learning and Language Modeling},
  author  = {Feng, Xidong and Luo, Yicheng and Wang, Ziyan and Tang, Hongrui and Yang, Mengyue and Shao, Kun and Mguni, David and Du, Yali and Wang, Jun},
  journal = {arXiv preprint arXiv:2306.09200},
  year    = {2023}
}
```
