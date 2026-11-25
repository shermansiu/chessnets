"""Lc0 policy/value inference pipeline."""

from __future__ import annotations

import pathlib
import typing as tp

import torch
import torch.nn.functional as F
import torch.utils.data

from chessnets.utils import board_history, decode_policy

from ..base import Pipeline, PipelineInputConfig, extract_domain_inputs
from ..outputs import ChessMovePrediction, ChessMovePredictionDataset
from ..pipeline_utils import (
    progress,
    DEFAULT_PARALLEL_THRESHOLD,
    adaptive_process_map,
    check_position_is_playable,
    iter_dataset_batches,
    resolve_num_workers,
)


def _prepare_lc0_sample(pgn: str):
    history = board_history.BoardHistory.from_pgn(pgn)
    return history, history.asarray()


class Lc0Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pgns: list[str] | str,
        *,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
        prepared = adaptive_process_map(
            _prepare_lc0_sample,
            pgns,
            parallel_threshold=parallel_threshold,
            max_workers=num_workers,
            chunksize=16,
            desc="Converting PGNs to board histories",
            leave=False,
        )
        self.histories = [history for history, _ in prepared]
        self.board_arrays = [board_array for _, board_array in prepared]

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, idx):
        history = self.histories[idx]
        board_array = self.board_arrays[idx]
        return (
            torch.from_numpy(board_array).reshape(
                112,
                64,  # Reshape to (channels, height * width)
            ),
            history,
        )

    @staticmethod
    def collate_fn(batch):
        board_tensors, histories = zip(*batch)
        board_tensor = torch.stack(board_tensors)
        return board_tensor, list(histories)


class _IndexedLc0Dataset(torch.utils.data.Dataset):
    """Attach source indices without changing the public ``Lc0Dataset`` API."""

    def __init__(self, dataset: Lc0Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return idx, self.dataset[idx]

    @staticmethod
    def collate_fn(batch):
        indices, samples = zip(*batch)
        board_tensor, histories = Lc0Dataset.collate_fn(samples)
        return list(indices), board_tensor, histories


GAME_OUTCOMES = ["1-0", "1/2-1/2", "0-1"]


class Lc0Pipeline(Pipeline):
    # Lc0 and Maia-1 networks are small enough for large batches.
    auto_batch_sizes = {"cpu": 256, "cuda": 1024}

    def __init__(
        self,
        model: torch.nn.Module,
        temperature: float = 1.0,
        input_config: PipelineInputConfig | None = None,
    ):
        super().__init__(input_config=input_config)
        self.model = model
        self.model.eval()
        self.temperature = max(1e-9, temperature)
        self.should_output_value = True
        self.should_output_moves_left = True

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(required=("pgns",))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path,
        device: tp.Any = "cpu",
        *,
        revision: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        repo_type: str | None = None,
        **kwargs: tp.Any,
    ) -> tp.Self:
        """Load an Lc0 checkpoint from Hugging Face or a local directory."""
        from chessnets.models.lc0 import AutoLc0Model

        loader_kwargs = {
            name: value
            for name, value in {
                "revision": revision,
                "cache_dir": cache_dir,
                "token": token,
                "local_files_only": local_files_only,
                "repo_type": repo_type,
            }.items()
            if value is not None and (name != "local_files_only" or value)
        }
        model = AutoLc0Model.from_pretrained(
            pretrained_model_name_or_path,
            is_eval=True,
            device=device,
            **loader_kwargs,
        )
        model.to(device=device)
        return cls(model, **kwargs)

    @torch.inference_mode(True)
    def forward(
        self,
        board_tensor: torch.Tensor,
        *args,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        """Get the policy from the model for the given board tensor."""
        out = self.model(
            board_tensor.to(
                device=self.model.device,
                dtype=self.model.dtype,
                non_blocking=board_tensor.is_pinned(),
            )
        )
        del board_tensor  # Free memory
        policy = out.policy.cpu()

        if self.should_output_value:
            if hasattr(out, "value"):
                value = out.value
            else:
                value = out.value_winner

            if value.shape[1] == 3:
                value = F.softmax(value, dim=1)
                result = value.cpu()
                value = (
                    value[:, 0] - value[:, 2]
                )  # Take expectation over WDL probabilities to get a single value
            else:
                result = None
            value = value.cpu()
        else:
            result = None
            value = None

        if (
            hasattr(out, "moves_left")
            and self.should_output_moves_left
            and out.moves_left is not None
        ):
            moves_left = out.moves_left.cpu()
        else:
            moves_left = None

        del out
        return policy, result, value, moves_left

    @staticmethod
    def post_process_predictions(
        policy: torch.Tensor,
        result: torch.Tensor | None,
        value: torch.Tensor | None,
        moves_left: torch.Tensor | None,
        history: board_history.BoardHistory,
        temperature: float,
        return_histories: bool = False,
    ):
        check_position_is_playable(history.active_board)
        position = history.positions[-1]
        # Pair each legal move with its policy index (from the side to move's
        # perspective), in index order. Keeping the board's own Move objects
        # avoids decoding UCI strings and preserves knight underpromotions,
        # which share their policy index with the plain pawn push.
        indexed_moves = sorted(
            (
                (decode_policy.get_move_index(move, position.is_black_turn), move)
                for move in position.board.legal_moves
            ),
            key=lambda pair: pair[0],
        )
        indices = torch.tensor([index for index, _ in indexed_moves], dtype=torch.long)
        probs = (policy[indices] / temperature).softmax(-1)

        preds_list = [
            (move, prob) for (_, move), prob in zip(indexed_moves, probs.tolist())
        ]
        best_move = preds_list[probs.argmax().item()][0]
        preds_list.sort(key=lambda x: x[1], reverse=True)

        best_move_san = history.active_board.san(best_move)
        if return_histories:
            history.append(best_move)

        return ChessMovePrediction(
            board_history=history if return_histories else None,
            best_move_san=best_move_san,
            best_move_uci=str(best_move),
            preds_list=dict(preds_list),
            win_probability=(
                (value.item() + 1) / 2 if value is not None else None
            ),  # Convert from [-1, 1] to [0, 1]
            result_probs=(
                {outcome: prob for outcome, prob in zip(GAME_OUTCOMES, result.tolist())}
                if result is not None
                else None
            ),
            result=(
                GAME_OUTCOMES[result.argmax().item()] if result is not None else None
            ),
            moves_left=moves_left.item() if moves_left is not None else None,
        )

    def make_prediction(
        self,
        pgns: str | list[str],
        *args,
        batch_size: int = 16,
        return_histories: bool = False,
        tqdm_leave: bool = False,
        shuffle: bool = False,
        persistent_workers: bool = False,  # Retained for API compatibility; LC0 batches eagerly.
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        """Make a prediction based on the provided PGN string.

        Returns a tuple of the board history, the best move in SAN format, and a dictionary
        of all legal moves with their probabilities.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.input_config.validate(pgns=pgns, **extract_domain_inputs(kwargs))
        pgn_count = 1 if isinstance(pgns, str) else len(pgns)
        resolved_workers = resolve_num_workers(
            pgn_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )
        dataset = Lc0Dataset(
            pgns=pgns,
            num_workers=resolved_workers,
            parallel_threshold=parallel_threshold,
        )
        is_using_cuda = torch.cuda.is_available() and self.model.device.type == "cuda"
        indexed_dataset = _IndexedLc0Dataset(dataset)
        predictions: list[ChessMovePrediction | None] = [None] * len(dataset)
        for indices, board_tensor, histories in progress(
            iter_dataset_batches(
                indexed_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=indexed_dataset.collate_fn,
                # Lc0 samples are prepared eagerly above, so worker processes
                # would only fetch already-materialized tensors.
                num_workers=0,
                pin_memory=is_using_cuda,
                # There are no persistent workers in the eager LC0 path.
                persistent_workers=False,
            ),
            desc="Making predictions",
            total=(len(dataset) + batch_size - 1) // batch_size,
            leave=tqdm_leave,
        ):
            policy, result, value, moves_left = self.forward(board_tensor, *args)
            if result is None:
                result = [None] * len(histories)
            if value is None:
                value = [None] * len(histories)
            if moves_left is None:
                moves_left = [None] * len(histories)
            raw_outputs = []
            for p, r, v, ml, history in zip(
                policy, result, value, moves_left, histories
            ):
                raw_outputs.append(
                    (
                        p,
                        r,
                        v,
                        ml,
                        history,
                        self.temperature,
                        return_histories,
                    )
                )
            # Keep post-processing within this batch. Starting a fresh process
            # pool per batch costs more than this small legal-move calculation.
            batch_predictions = list(
                map(self.post_process_predictions, *zip(*raw_outputs))
            )
            for index, prediction in zip(indices, batch_predictions):
                predictions[index] = prediction
        # Every dataset item is visited exactly once, including when shuffled.
        completed_predictions = tp.cast(list[ChessMovePrediction], predictions)
        return ChessMovePredictionDataset[ChessMovePrediction](
            predictions=completed_predictions
        )


__all__ = ["GAME_OUTCOMES", "Lc0Dataset", "Lc0Pipeline"]
