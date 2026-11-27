"""Dependency-light helpers shared by inference pipelines."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import random
import typing as tp
import warnings

import chess
import torch

CPU_COUNT = os.cpu_count()
PREPROCESSING_WORKERS = min(4, CPU_COUNT) if CPU_COUNT is not None else 1
POSTPROCESSING_WORKERS = (
    CPU_COUNT - PREPROCESSING_WORKERS if CPU_COUNT is not None else 1
)
DEFAULT_PARALLEL_THRESHOLD = 32
T = tp.TypeVar("T")


def progress(iterable: tp.Iterable[T], **kwargs: tp.Any) -> tp.Iterator[T]:
    """Iterate with a tqdm progress bar when tqdm is installed.

    Log records emitted while iterating are routed through tqdm so they do not
    break the progress bar. Without tqdm, the iterable is yielded unchanged.
    """
    try:
        from tqdm.auto import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm
    except ModuleNotFoundError:
        yield from iterable
        return
    with logging_redirect_tqdm():
        yield from tqdm(iterable, **kwargs)


class DrawnPositionWarning(UserWarning):
    """A move was requested for a position that is already drawn by rule.

    Benchmarks built from real games can contain such positions, because
    players may keep playing after a threefold repetition (or online servers do
    not end the game), so a prediction is still made. Filter this category with
    ``warnings.simplefilter("ignore", DrawnPositionWarning)`` to silence it.
    """


def check_position_is_playable(board: chess.Board) -> None:
    """Raise for checkmate or stalemate; warn if the position is drawn by rule.

    Repetition is detected from ``board``'s move stack, so a board created from
    a bare FEN can only report the move-count and material rules.
    """
    if not any(board.generate_legal_moves()):
        ending = "checkmate" if board.is_check() else "stalemate"
        raise ValueError(
            f"Cannot predict a move: the game is over by {ending} ({board.fen()})"
        )
    reasons = []
    if board.is_repetition(5):
        reasons.append("fivefold repetition")
    elif board.is_repetition(3):
        reasons.append("threefold repetition")
    if board.is_seventyfive_moves():
        reasons.append("the 75-move rule")
    elif board.is_fifty_moves():
        reasons.append("the 50-move rule")
    if board.is_insufficient_material():
        reasons.append("insufficient material")
    if reasons:
        warnings.warn(
            f"Predicting a move in a position already drawn by {' and '.join(reasons)}: "
            f"{board.fen()}",
            DrawnPositionWarning,
            stacklevel=3,
        )


def mirror_move(move: chess.Move):
    return chess.Move(
        from_square=chess.square_mirror(move.from_square),
        to_square=chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


def free_memory(collect_garbage: bool = False):
    if collect_garbage:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_out_of_memory_error(error: BaseException) -> bool:
    """Whether ``error``, or an error it was raised from, is an out-of-memory error."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
            return True
        if isinstance(error, RuntimeError) and "out of memory" in str(error):
            return True
        error = error.__cause__ or error.__context__
    return False


def process_map(function, *iterables, **kwargs):
    try:
        from tqdm.contrib.concurrent import process_map as _process_map
    except ModuleNotFoundError:
        return list(map(function, *iterables))
    return _process_map(function, *iterables, **kwargs)


def resolve_num_workers(
    item_count: int,
    requested_workers: int | None = None,
    parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
) -> int:
    """Choose a worker count without starting processes for small requests.

    Dataset and process-pool startup is often more expensive than model
    inference for one or a few positions.  ``None`` therefore means adaptive
    selection, while an explicit value always takes precedence.
    """
    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    if parallel_threshold < 0:
        raise ValueError("parallel_threshold must be non-negative")
    if requested_workers is not None and requested_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if requested_workers is not None:
        return requested_workers
    if item_count <= parallel_threshold:
        return 0
    return PREPROCESSING_WORKERS


def iter_dataset_batches(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    collate_fn,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
):
    """Yield dataset batches with a cheap single-process path.

    The synchronous path deliberately avoids constructing a DataLoader.  This
    is useful for inference requests that contain only a handful of examples;
    the parallel path retains DataLoader's normal worker and collation
    behavior for larger requests.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")

    item_count = len(dataset)
    if num_workers == 0:
        indices = list(range(item_count))
        if shuffle:
            random.shuffle(indices)
        for start in range(0, item_count, batch_size):
            batch_indices = indices[start : start + batch_size]
            if batch_indices:
                yield collate_fn([dataset[index] for index in batch_indices])
        return

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    yield from dataloader


def adaptive_process_map(
    function,
    *iterables,
    parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    max_workers: int | None = None,
    **kwargs,
):
    """Map inline for small inputs and use :func:`process_map` for large ones."""
    if parallel_threshold < 0:
        raise ValueError("parallel_threshold must be non-negative")
    if max_workers is not None and max_workers < 0:
        raise ValueError("max_workers must be non-negative")
    if not iterables:
        return []

    iterables = tuple(iterables)
    try:
        item_count = len(iterables[0])
    except TypeError:
        iterables = tuple(tuple(iterable) for iterable in iterables)
        item_count = len(iterables[0])

    if item_count == 0:
        return []
    if max_workers == 0 or (max_workers is None and item_count <= parallel_threshold):
        return list(map(function, *iterables))

    if max_workers is not None:
        kwargs["max_workers"] = max_workers
    return process_map(function, *iterables, **kwargs)


@contextlib.contextmanager
def pause_running_loop() -> tp.Iterator[None]:
    original_loop = asyncio._get_running_loop()
    asyncio._set_running_loop(None)
    try:
        yield
    finally:
        asyncio._set_running_loop(original_loop)


def asyncio_run(coroutine: tp.Coroutine[tp.Any, tp.Any, T]) -> T:
    if asyncio._get_running_loop() is not None:
        with pause_running_loop():
            return asyncio.run(coroutine)
    return asyncio.run(coroutine)


__all__ = [
    "CPU_COUNT",
    "DEFAULT_PARALLEL_THRESHOLD",
    "POSTPROCESSING_WORKERS",
    "PREPROCESSING_WORKERS",
    "DrawnPositionWarning",
    "adaptive_process_map",
    "asyncio_run",
    "check_position_is_playable",
    "free_memory",
    "iter_dataset_batches",
    "mirror_move",
    "pause_running_loop",
    "process_map",
    "resolve_num_workers",
]
