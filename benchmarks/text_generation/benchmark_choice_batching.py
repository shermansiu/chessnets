"""Compare batched and unbatched ``constraint="choice"`` generation.

Runs a text-generation profile on positions from a CSV file, first one
position at a time (the reference), then with ``batched_choice=True`` at each
batch size, with and without ``sort_by_length=True``. For each run it reports
the time per position and how many moves match the reference.

With ``--explain``, it also traces every position whose move differs: it
regenerates the reference with the scores kept and prints the log
probabilities of both answers' tokens where they part. Differences of 0 or one
bfloat16 step are ties that padding tipped the other way.

The CSV needs ``pgn`` and ``prompt`` columns (``prompt`` may be omitted for
formats that don't use one), such as a ChessEval ``full_results.csv``. With a
``raw_response`` column, it also reports how many reference answers match it.

    python benchmarks/text_generation/benchmark_choice_batching.py \\
        --csv full_results.csv --sample 256 --batch-sizes 8 16 32 --explain
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time
import warnings

import torch

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chessnets import AutoPipeline  # noqa: E402 (after sys.path setup)


def _load_positions(path: str, sample: int | None, seed: int):
    import pandas as pd

    frame = pd.read_csv(path)
    if sample is not None and sample < len(frame):
        frame = frame.sample(sample, random_state=seed)
    return frame.reset_index(drop=True)


def _run(pipeline, frame, batch_size: int, **options):
    pipeline.pipeline_config = dataclasses.replace(pipeline.pipeline_config, **options)
    inputs = {"prompt": frame.prompt.tolist()} if "prompt" in frame else {}
    pgns = frame.pgn.tolist()
    # Warm up on one batch, so compilation and allocation aren't timed.
    pipeline.make_prediction(
        pgns[:batch_size],
        **{k: v[:batch_size] for k, v in inputs.items()},
        batch_size=batch_size,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = pipeline.make_prediction(pgns, **inputs, batch_size=batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start, result.predictions


def _explain(pipeline, frame, index: int, reference: str, other: str) -> str:
    """Compare both answers' log probabilities where they part."""
    import outlines
    import transformers
    from outlines_core import Index

    from chessnets.pipelines.text_generation.batched_choice import (
        PerRowChoiceLogitsProcessor,
    )
    from chessnets.pipelines.text_generation.pipeline_text_generation import (
        _PromptDataset,
    )

    dataset = _PromptDataset(
        pgns=[frame.pgn[index]],
        prompts=[frame.prompt[index]],
        should_shuffle_legal_moves=False,
        prompt_template=pipeline.prompt_template,
        num_workers=0,
    )
    choices = pipeline.legal_move_choices(dataset.histories[0], dataset.legal_moves[0])
    vocabulary = pipeline._choice_backend_for_model().vocabulary
    processor = PerRowChoiceLogitsProcessor(
        [Index(outlines.types.dsl.to_regex(outlines.types.Choice(choices)), vocabulary)]
    )
    model, tokenizer = pipeline.model.model, pipeline.model.hf_tokenizer
    encoded = tokenizer([dataset.consolidated_prompts[0]], return_tensors="pt").to(
        model.device
    )
    output = model.generate(
        **encoded,
        logits_processor=transformers.LogitsProcessorList([processor]),
        max_new_tokens=16,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output.sequences[0, encoded.input_ids.shape[1] :].tolist()
    # The other answer parted at the first step where a token fits it but not
    # this answer. Tokens can split differently (" Bd" or " B" + "d"), so a
    # prefix both answers share doesn't mean the tokens were the same.
    text = ""
    for step, token in enumerate(generated):
        log_probs = torch.log_softmax(output.scores[step][0].float(), -1)
        candidates = []
        for t in torch.topk(log_probs, 20).indices.tolist():
            piece = text + tokenizer.decode([t])
            if (
                t != token
                and other.startswith(piece)
                and not reference.startswith(piece)
            ):
                candidates.append(t)
        if candidates:
            chosen, alternative = (
                log_probs[token].item(),
                log_probs[candidates[0]].item(),
            )
            return (
                f"step {step}: {tokenizer.decode([token])!r} {chosen:.4f} vs "
                f"{tokenizer.decode([candidates[0]])!r} {alternative:.4f} "
                f"(gap {chosen - alternative:.4f})"
            )
        text += tokenizer.decode([token])
        if not other.startswith(text):
            return f"step {step}: the other answer's token is not in the top 20"
    return "the answers don't part within the generated tokens"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", required=True, help="positions: pgn, prompt columns")
    parser.add_argument("--model", default="Waterhorse/chessgpt-base-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--no-sorted", action="store_true", help="skip sorted runs")
    parser.add_argument("--explain", action="store_true", help="trace differences")
    args = parser.parse_args()

    warnings.simplefilter("ignore")
    frame = _load_positions(args.csv, args.sample, args.seed)
    kwargs = {"constraint": "choice"}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    pipeline = AutoPipeline.from_pretrained(args.model, device=args.device, **kwargs)
    count = len(frame)

    reference_seconds, reference = _run(
        pipeline, frame, 1, batched_choice=False, sort_by_length=False
    )
    print(f"{count} positions, {args.model}, {torch.__version__}")
    print("| Run | ms/position | Speedup | Same move as unbatched |")
    print("|---|---|---|---|")
    print(f"| Unbatched | {reference_seconds / count * 1000:.0f} | 1.0x | - |")
    differing: dict[int, str] = {}
    for sort in [False] if args.no_sorted else [False, True]:
        for batch_size in args.batch_sizes:
            seconds, predictions = _run(
                pipeline, frame, batch_size, batched_choice=True, sort_by_length=sort
            )
            same = 0
            for index, (a, b) in enumerate(zip(reference, predictions)):
                if a.best_move_uci == b.best_move_uci:
                    same += 1
                else:
                    differing.setdefault(index, b.raw_response or "")
            label = f"Batched{', sorted' if sort else ''}, bs={batch_size}"
            print(
                f"| {label} | {seconds / count * 1000:.0f} | "
                f"{reference_seconds / seconds:.1f}x | {same}/{count} |"
            )
    if "raw_response" in frame:
        archived = frame.raw_response.astype(str).str.strip().tolist()
        matches = sum(
            a == (p.raw_response or "").strip() for a, p in zip(archived, reference)
        )
        print(f"\nUnbatched answers equal to the CSV's raw_response: {matches}/{count}")
    if args.explain and differing:
        print(f"\n{len(differing)} positions differ in some run:")
        for index, other in sorted(differing.items()):
            ours = reference[index].raw_response or ""
            print(f"- {index}: {ours!r} vs {other!r}: ", end="")
            print(_explain(pipeline, frame, index, ours, other))


if __name__ == "__main__":
    main()
