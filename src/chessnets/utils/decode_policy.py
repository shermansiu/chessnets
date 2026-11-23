"""Lc0 policy move indexing and move-orientation helpers."""

from __future__ import annotations

import chess


def _is_standard_policy_move(source: int, target: int) -> bool:
    if source == target:
        return False
    dx = abs(chess.square_file(target) - chess.square_file(source))
    dy = abs(chess.square_rank(target) - chess.square_rank(source))
    return dx == 0 or dy == 0 or dx == dy or sorted((dx, dy)) == [1, 2]


def _make_policy_moves() -> tuple[str, ...]:
    moves = [
        chess.square_name(source) + chess.square_name(target)
        for source in chess.SQUARES
        for target in chess.SQUARES
        if _is_standard_policy_move(source, target)
    ]
    for source_file in range(8):
        source = chess.square(source_file, 6)
        for target_file in range(max(0, source_file - 1), min(7, source_file + 1) + 1):
            target = chess.square(target_file, 7)
            for promotion in "qrb":
                moves.append(
                    chess.square_name(source) + chess.square_name(target) + promotion
                )
    if len(moves) != 1858:
        raise RuntimeError(f"Generated an invalid Lc0 policy size: {len(moves)}")
    return tuple(moves)


MOVES = _make_policy_moves()
MOVE_TO_INDEX = {move: idx for idx, move in enumerate(MOVES)}


def get_move_index(move: chess.Move, should_mirror: bool = False) -> int:
    """Get the index of a chess move."""
    if should_mirror:
        move = mirror_move(move)
    uci = move.uci()
    if uci.endswith("n"):  # N promotion is encoded as a normal move
        uci = uci.removesuffix("n")
    return MOVE_TO_INDEX[uci]


def mirror_move(move: chess.Move) -> chess.Move:
    """Mirror a move vertically."""
    return chess.Move(
        from_square=chess.square_mirror(move.from_square),
        to_square=chess.square_mirror(move.to_square),
        promotion=move.promotion,
        drop=move.drop,
    )


__all__ = ["MOVES", "MOVE_TO_INDEX", "get_move_index", "mirror_move"]
