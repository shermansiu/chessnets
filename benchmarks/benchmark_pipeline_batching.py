"""Measure adaptive batching overhead without loading a chess model.

This benchmark isolates the CPU-side costs that matter for small inference
requests. It intentionally uses a tiny synthetic dataset so that process-pool
and DataLoader startup are visible in the timings.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chessnets.pipelines.pipeline_utils import (  # noqa: E402 (after sys.path setup)
    DEFAULT_PARALLEL_THRESHOLD,
    adaptive_process_map,
    iter_dataset_batches,
    resolve_num_workers,
)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self.values = list(range(size))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def collate(batch):
    # Keep the benchmark focused on worker startup and batching. Returning
    # plain values avoids measuring torch's shared-memory transport as well.
    return list(batch)


def square(value: int) -> int:
    return value * value


def benchmark_size(
    size: int,
    *,
    batch_size: int,
    parallel_threshold: int,
    requested_workers: int | None,
) -> tuple[int, float, float, float]:
    workers = resolve_num_workers(
        size,
        requested_workers=requested_workers,
        parallel_threshold=parallel_threshold,
    )
    dataset = SyntheticDataset(size)

    start = time.perf_counter()
    values = []
    for batch in iter_dataset_batches(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=workers,
    ):
        values.extend(batch)
    batch_seconds = time.perf_counter() - start

    start = time.perf_counter()
    adaptive_process_map(
        square,
        values,
        parallel_threshold=parallel_threshold,
        max_workers=workers,
        disable=True,
    )
    post_seconds = time.perf_counter() - start
    return workers, batch_seconds, post_seconds, batch_seconds + post_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 32, 64, 256],
        help="request sizes to measure",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--parallel-threshold",
        type=int,
        default=DEFAULT_PARALLEL_THRESHOLD,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="force a worker count; omit for adaptive selection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.parallel_threshold < 0:
        raise SystemExit("--parallel-threshold must be non-negative")
    if any(size < 0 for size in args.sizes):
        raise SystemExit("--sizes must be non-negative")
    if args.workers is not None and args.workers < 0:
        raise SystemExit("--workers must be non-negative")

    print("size,workers,batching_seconds,postprocessing_seconds,total_seconds")
    for size in args.sizes:
        workers, batch_seconds, post_seconds, total_seconds = benchmark_size(
            size,
            batch_size=args.batch_size,
            parallel_threshold=args.parallel_threshold,
            requested_workers=args.workers,
        )
        print(
            f"{size},{workers},{batch_seconds:.6f},{post_seconds:.6f},"
            f"{total_seconds:.6f}"
        )


if __name__ == "__main__":
    main()
