# SPDX-License-Identifier: MIT
# Portions adapted from Maia 2 (https://github.com/CSSLab/maia2),
# Copyright (c) 2024 CSSLab. See LICENSES/Maia2-MIT.txt.
# Modified for ChessNets by Sherman Siu; modifications are under the same license.

import chess


def create_elo_dict() -> dict[str, int]:
    inteval = 100
    start = 1100
    end = 2000

    range_dict = {f"<{start}": 0}
    range_index = 1

    for lower_bound in range(start, end - 1, inteval):
        upper_bound = lower_bound + inteval
        range_dict[f"{lower_bound}-{upper_bound - 1}"] = range_index
        range_index += 1

    range_dict[f">={end}"] = range_index

    # print(range_dict, flush=True)

    return range_dict


def generate_pawn_promotions() -> list[str]:
    # Define the promotion rows for both colors and the promotion pieces
    # promotion_rows = {'white': '7', 'black': '2'}
    promotion_rows = {"white": "7"}
    promotion_pieces = ["q", "r", "b", "n"]
    promotions = []

    # Iterate over each color
    for color, row in promotion_rows.items():
        # Target rows for promotion (8 for white, 1 for black)
        target_row = "8" if color == "white" else "1"

        # Each file from 'a' to 'h'
        for file in "abcdefgh":
            # Direct move to promotion
            for piece in promotion_pieces:
                promotions.append(f"{file}{row}{file}{target_row}{piece}")

            # Capturing moves to the left and right (if not on the edges of the board)
            if file != "a":
                left_file = chr(ord(file) - 1)  # File to the left
                for piece in promotion_pieces:
                    promotions.append(f"{file}{row}{left_file}{target_row}{piece}")

            if file != "h":
                right_file = chr(ord(file) + 1)  # File to the right
                for piece in promotion_pieces:
                    promotions.append(f"{file}{row}{right_file}{target_row}{piece}")

    return promotions


def get_all_possible_moves() -> list[str]:
    all_moves = []

    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)

            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)

            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)

    all_moves: list[str] = [all_moves[i].uci() for i in range(len(all_moves))]

    pawn_promotions = generate_pawn_promotions()

    return all_moves + pawn_promotions


MOVES = get_all_possible_moves()
MOVE_TO_INDEX = {move: idx for idx, move in enumerate(MOVES)}
ELO_DICT = create_elo_dict()


def map_to_category(elo: int) -> int:
    interval = 100
    start = 1100
    end = 2000

    if elo < start:
        return ELO_DICT[f"<{start}"]
    elif elo >= end:
        return ELO_DICT[f">={end}"]
    else:
        for lower_bound in range(start, end - 1, interval):
            upper_bound = lower_bound + interval
            if lower_bound <= elo < upper_bound:
                return ELO_DICT[f"{lower_bound}-{upper_bound - 1}"]
    raise ValueError(f"ELO {elo} is out of range.")
