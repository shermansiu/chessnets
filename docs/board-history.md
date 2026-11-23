# Working with board histories

`BoardHistory` stores one mutable active board and a sequence of stackless
position snapshots. This keeps historical access and Lc0 input generation fast
without retaining a second move stack inside every historical `chess.Board`.

This guide covers copying histories, building model inputs, evaluating
candidate moves, and keeping many games in memory efficiently.

## Storage model and ownership

`BoardHistory` relies on these ownership rules:

| Value | Purpose | Mutation rule |
| --- | --- | --- |
| `active_board` | Current position and the move stack used to reach it | Mutated by `BoardHistory.append()` |
| `moves` | ChessNets move sequence and training targets | Mutated by `BoardHistory.append()` |
| `positions` | Stackless `LeelaBoard` snapshots, including the initial position | Treat every snapshot and its board as immutable |
| `last_seen_table` | References to snapshots used for repetition metadata | Managed by `BoardHistory` |

Each call to `append()` creates a board with `copy(stack=False)`. The snapshot
contains the bitboards, turn, castling rights, en-passant square, and clocks,
but its `move_stack` is empty. `active_board` remains the place to query the
current legal moves, generate SAN, or continue the game.

Do not push, pop, or otherwise mutate `history.positions[i].board`. Forks share
those snapshots, so mutating one can corrupt multiple histories. Create a fork
or operate on `active_board` through `BoardHistory` instead.

## Basic construction

For a complete PGN, let `BoardHistory` parse and append the moves once:

```python
from chessnets.utils.board_history import BoardHistory

history = BoardHistory.from_pgn(pgn)
board_array = history.asarray()
```

For an online or generated game, keep one history and append moves incrementally:

```python
import chess

from chessnets.utils.board_history import BoardHistory

history = BoardHistory()
move = chess.Move.from_uci("e2e4")

if move not in history.active_board.legal_moves:
    raise ValueError(f"Illegal move: {move}")

history.append(move)
board_array = history.asarray()
```

`python-chess` does not require `Board.push()` to receive a legal move, so code
accepting untrusted or generated moves should validate them before `append()`.

To begin at an arbitrary position, pass a board explicitly:

```python
history = BoardHistory(active_board=chess.Board(fen))
```

That position becomes index `0` of this history. Earlier moves are not added to
`history.moves`; use `from_pgn()` when the preceding move sequence matters.

## Copying histories

### Append a candidate move without changing the source

Avoid deep-copying a history, which duplicates every board and container:

```python
import copy

candidate = copy.deepcopy(history)
candidate.append(move)
```

Use `copy_with_move()` instead:

```python
candidate = history.copy_with_move(move)
```

`copy_with_move()` copies the mutable active board, move list, repetition
tables, and list containers. It shares the stackless historical snapshots and
then appends one new snapshot. Changes to the candidate's mutable containers do
not affect the source history.

This is the pattern used when ChessNets scores every legal move:

```python
candidate_arrays = []

for move in history.active_board.legal_moves:
    candidate = history.copy_with_move(move)
    candidate_arrays.append(candidate.asarray())
```

### Continue along an independent branch

When several moves will be appended to the branch, fork once:

```python
branch = history.fork()

for move in continuation:
    branch.append(move)
```

Do not call `copy_with_move()` repeatedly on the original history to build one
continuation. That recopying is unnecessary; fork once and append to the fork.

### Copying a bare `chess.Board`

When code outside `BoardHistory` needs only a position snapshot, make the stack
choice explicit:

```python
position_only = board.copy(stack=False)
independent_game = board.copy(stack=True)
```

Avoid `copy.copy(board)` and `copy.deepcopy(board)`. Their stack behavior is
less clear at the call site, and retaining the stack is expensive when only the
current position is needed.

## Efficient model-input generation

### Current or historical Lc0 input

Use `asarray()` rather than reconstructing positions by replaying moves:

```python
current = history.asarray()
earlier = history.asarray(last_position_index=25)
```

Indices address `positions`, not `moves`: position `0` is the initial position,
and position `i + 1` is the board after move `i`. Negative indices follow the
usual Python convention. Out-of-range indices raise `IndexError`.

`asarray()` reads at most eight snapshots regardless of total game length. It
returns a new `(112, 64)` NumPy array, so compute it once and reuse the result
when several consumers need the same input:

```python
board_array = history.asarray()
board_tensor = torch.from_numpy(board_array).reshape(112, 8, 8)
```

Avoid re-parsing prefixes of the same game:

```python
# Replays an increasing prefix for every position.
arrays = [BoardHistory.from_pgn(prefix).asarray() for prefix in pgn_prefixes]
```

Parse once and address each historical position:

```python
history = BoardHistory.from_pgn(pgn)
arrays = [history.asarray(i) for i in requested_position_indices]
```

This still allocates one model array per requested position. For large datasets,
generate batches lazily instead of storing every array for every game.

### Maia2 input

Maia2 uses only one position rather than the eight-position Lc0 history:

```python
current = history.to_maia2_array()
earlier = history.to_maia2_array(position_index=25)
```

If no game history is available or required, avoid creating a synthetic
`BoardHistory`:

```python
from chessnets.utils.board_history import LeelaBoard

board_array = LeelaBoard.from_fen(fen).to_maia2_array()
```

## Candidate-move scaling

`copy_with_move()` is the safe general-purpose API and is much cheaper than
deep-copying a history. It is not constant-time in game length:
`fork()` copies the active move stack, move list, positions list, and repetition
maps. The `LeelaBoard` snapshots themselves are shared.

For `b` candidate moves in a history of length `n`, retaining all candidate
histories therefore uses O(`b * n`) container and active-stack space. Prefer a
streaming loop when only arrays or scores must survive:

```python
for move in history.active_board.legal_moves:
    candidate = history.copy_with_move(move)
    board_array = candidate.asarray()
    yield move, score(board_array)
```

Do not save each `candidate` unless downstream code genuinely needs the full
branch. Saving only the move and model output lets each candidate fork be
released after scoring.

## Bulk and parallel workflows

- Parse each PGN once. Reuse its `BoardHistory` for all requested positions.
- Convert arrays to tensors once per sample, then batch tensors before model
  inference.
- Stream samples or batches when the complete array dataset does not need to
  reside in memory.
- Send independent histories to worker processes. Do not concurrently append
  to one shared `BoardHistory`.
- Fork before continuing the same source position along independent branches.
- Keep model outputs, moves, or FENs instead of full candidate histories when
  the histories are not needed later.

The ChessNets dataset pipelines use process-based mapping when they parse many
independent PGNs. Follow the same pattern for CPU-heavy preprocessing, but use a
coarse enough chunk size that process serialization does not dominate the board
encoding work.

## Complexity reference

Here, `n` is the number of moves in a history and `u` is the number of distinct
position keys retained for repetition tracking.

| Operation | Time | Additional space | Notes |
| --- | ---: | ---: | --- |
| `append(move)` | O(1) | O(1) | Chess has a fixed 64-square state; the new snapshot has no move stack |
| `asarray(index)` | O(1) | O(1) | Reads at most eight positions and allocates a fixed-size array |
| Entire `BoardHistory` | — | O(`n + u`) | One active stack plus one stackless snapshot per position |
| `fork()` | O(`n + u`) | O(`n + u`) | Copies mutable containers and active stack; shares snapshots |
| `copy_with_move(move)` | O(`n + u`) | O(`n + u`) | One fork plus one append |
| `deepcopy(history)` | O(`n + u`) | O(`n + u`) | Also duplicates every snapshot; avoid it |
