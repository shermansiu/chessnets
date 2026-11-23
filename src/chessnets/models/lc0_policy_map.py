"""Deterministic mappings between Lc0 policy-head layouts and UCI moves.

The mappings are derived from the documented tensor layouts. They are model
structure, not learned checkpoint state, so callers should register them as
non-persistent buffers.
"""

from __future__ import annotations

import chess
import numpy as np

from chessnets.utils.decode_policy import MOVES


_QUEEN_DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
_KNIGHT_DIRECTIONS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)


def _delta(uci: str) -> tuple[int, int, int]:
    source = chess.parse_square(uci[:2])
    target = chess.parse_square(uci[2:4])
    return (
        chess.square_file(target) - chess.square_file(source),
        chess.square_rank(target) - chess.square_rank(source),
        source,
    )


def convolution_policy_indices() -> np.ndarray:
    """Return the 80x8x8 flattened logit index for each policy move."""
    indices: list[int] = []
    for uci in MOVES:
        dx, dy, source = _delta(uci)
        if len(uci) == 5:
            direction = {-1: 0, 0: 1, 1: 2}[dx]
            promotion = {"r": 0, "b": 1, "q": 2}[uci[4]]
            plane = 64 + direction * 3 + promotion
        elif (dx, dy) in _KNIGHT_DIRECTIONS:
            plane = 56 + _KNIGHT_DIRECTIONS.index((dx, dy))
        else:
            step = max(abs(dx), abs(dy))
            direction = (dx // step, dy // step)
            plane = _QUEEN_DIRECTIONS.index(direction) * 7 + step - 1
        indices.append(plane * 64 + source)
    return np.asarray(indices, dtype=np.int64)


def attention_policy_indices() -> np.ndarray:
    """Return the flattened attention/promotion logit index per policy move."""
    indices: list[int] = []
    for uci in MOVES:
        source = chess.parse_square(uci[:2])
        target = chess.parse_square(uci[2:4])
        if len(uci) == 4:
            indices.append(source * 64 + target)
        else:
            source_file = chess.square_file(source)
            target_file = chess.square_file(target)
            promotion = {"q": 0, "r": 1, "b": 2}[uci[4]]
            indices.append(4096 + source_file * 24 + target_file * 3 + promotion)
    return np.asarray(indices, dtype=np.int64)


def arc_position_encoding() -> np.ndarray:
    """Encode queen/knight reachability for the 64 board squares."""
    encoding = np.zeros((1, 64, 64), dtype=np.float32)
    for source in chess.SQUARES:
        encoding[0, source, source] = -1.0
        source_file = chess.square_file(source)
        source_rank = chess.square_rank(source)
        for target in chess.SQUARES:
            if source == target:
                continue
            target_file = chess.square_file(target)
            target_rank = chess.square_rank(target)
            dx = target_file - source_file
            dy = target_rank - source_rank
            queen_reachable = dx == 0 or dy == 0 or abs(dx) == abs(dy)
            knight_reachable = sorted((abs(dx), abs(dy))) == [1, 2]
            if queen_reachable or knight_reachable:
                encoding[0, source, target] = 1.0
    return encoding


__all__ = [
    "arc_position_encoding",
    "attention_policy_indices",
    "convolution_policy_indices",
]
