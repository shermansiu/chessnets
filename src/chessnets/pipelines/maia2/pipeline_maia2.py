"""Maia2 inference pipeline."""

from __future__ import annotations

import typing as tp

import chess
import numpy as np
import torch
import torch.utils.data

from chessnets.models.maia2 import utils as maia2_utils
from chessnets.utils import board_history

if tp.TYPE_CHECKING:
    from chessnets.models.maia2 import Maia2Model

from ..base import Pipeline, PipelineInputConfig, extract_domain_inputs
from ..outputs import Maia2ChessMovePrediction, ChessMovePredictionDataset
from ..pipeline_utils import (
    progress,
    DEFAULT_PARALLEL_THRESHOLD,
    adaptive_process_map,
    check_position_is_playable,
    iter_dataset_batches,
    mirror_move,
    resolve_num_workers,
)


def _load_maia2_model() -> type[Maia2Model]:
    from chessnets.models.maia2 import Maia2Model

    return Maia2Model


class Maia2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fens: list[str] | str,
        elo_self: list[int] | int,
        elo_oppo: list[int] | int,
    ):
        if isinstance(fens, str):
            fens = [fens]
        if isinstance(elo_self, (np.int32, np.int64)):
            elo_self = [int(elo_self)]
        elif isinstance(elo_self, int):
            elo_self = [elo_self]
        if isinstance(elo_oppo, (np.int32, np.int64)):
            elo_oppo = [int(elo_oppo)]
        elif isinstance(elo_oppo, int):
            elo_oppo = [elo_oppo]
        assert len(fens) == len(elo_self) == len(elo_oppo), (
            "Length of FENs and ELOs must match."
        )
        self.fens = fens
        self.elo_self = elo_self
        self.elo_oppo = elo_oppo

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        elo_self = self.elo_self[idx]
        elo_oppo = self.elo_oppo[idx]
        lboard = board_history.LeelaBoard.from_fen(fen)
        board_array = lboard.to_maia2_array()  # Reshape to (channels, height, width)
        legal_moves_mask = lboard.to_maia2_legal_moves_mask()  # Reshape to (num_moves,)

        board_tensor = torch.from_numpy(board_array)
        del board_array  # Free memory

        elo_self_tensor = torch.tensor(
            maia2_utils.map_to_category(elo_self), dtype=torch.int32
        )
        elo_oppo_tensor = torch.tensor(
            maia2_utils.map_to_category(elo_oppo), dtype=torch.int32
        )
        return (
            board_tensor,
            elo_self_tensor,
            elo_oppo_tensor,
            torch.from_numpy(legal_moves_mask),
            lboard,
        )

    @staticmethod
    def collate_fn(batch):
        (
            board_tensors,
            elo_self_tensors,
            elo_oppo_tensors,
            legal_moves_masks,
            lboards,
        ) = zip(*batch)
        board_tensor = torch.stack(board_tensors)
        elo_self_tensor = torch.stack(elo_self_tensors)
        elo_oppo_tensor = torch.stack(elo_oppo_tensors)
        legal_moves_mask = torch.stack(legal_moves_masks)
        return (
            board_tensor,
            elo_self_tensor,
            elo_oppo_tensor,
            legal_moves_mask,
            list(lboards),
        )


class Maia2Pipeline(Pipeline):
    auto_batch_sizes = {"cpu": 256, "cuda": 1024}

    def __init__(
        self,
        model: Maia2Model,
        temperature: float = 1.0,
        input_config: PipelineInputConfig | None = None,
    ):
        super().__init__(input_config=input_config)
        self.model = model
        self.model.eval()
        self.temperature = max(1e-9, temperature)

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(
            required=("fens", "elo_self", "elo_opponent")
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | None = None,
        device: str | torch.device = "cpu",
        *,
        time_control: tp.Literal["blitz", "rapid"] | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        use_safetensors: bool | None = None,
        **kwargs: tp.Any,
    ) -> tp.Self:
        model_source = pretrained_model_name_or_path
        if pretrained_model_name_or_path is not None:
            aliases = {
                "maia2_blitz": "blitz",
                "maia2_rapid": "rapid",
                "maia2-blitz": "blitz",
                "maia2-rapid": "rapid",
                "blitz": "blitz",
                "rapid": "rapid",
            }
            selected_time_control = aliases.get(pretrained_model_name_or_path)
            if selected_time_control is not None:
                if time_control is not None and time_control != selected_time_control:
                    raise ValueError(
                        "pretrained_model_name_or_path and time_control select "
                        "different Maia2 checkpoints"
                    )
                time_control = selected_time_control
        if model_source is None:
            if time_control is None:
                raise TypeError(
                    "Specify a Maia 2 repository or time_control ('blitz'/'rapid')."
                )
            model_source = time_control
        model = _load_maia2_model().from_pretrained(
            model_source,
            time_control=time_control,
            device=device,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            use_safetensors=use_safetensors,
        )
        return cls(model, **kwargs)

    @torch.inference_mode(True)
    def forward(
        self,
        board_tensor: torch.Tensor,
        elo_self_tensor: torch.IntTensor,
        elo_oppo_tensor: torch.IntTensor,
        legal_moves_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Get the policy from the model for the given board tensor."""
        logits_maia, _, logits_value = self.model(
            board_tensor.to(
                device=self.model.device,
                dtype=self.model.dtype,
                non_blocking=board_tensor.is_pinned(),
            ),
            elo_self_tensor.to(device=self.model.device),
            elo_oppo_tensor.to(device=self.model.device),
        )
        legal_moves_mask = legal_moves_mask.to(device=self.model.device)
        logits_maia_legal = logits_maia * legal_moves_mask
        probs = (logits_maia_legal / self.temperature).softmax(dim=-1).cpu()
        logits_value = (logits_value / 2 + 0.5).clamp(0, 1).cpu()
        del (
            board_tensor,
            elo_self_tensor,
            elo_oppo_tensor,
            legal_moves_mask,
            logits_maia_legal,
        )
        return probs, logits_value

    @staticmethod
    def post_process_predictions(
        leela_board: board_history.LeelaBoard,
        legal_move_indices_tensor: torch.Tensor,
        probs: torch.Tensor,
        logits_value: torch.Tensor,
    ):
        check_position_is_playable(leela_board.board)
        is_black_turn = leela_board.is_black_turn

        legal_move_indices = legal_move_indices_tensor.nonzero().flatten().tolist()

        preds_list: list[tuple[chess.Move, float]] = []
        for legal_move_index in legal_move_indices:
            current_move_str = maia2_utils.MOVES[legal_move_index]
            current_move = chess.Move.from_uci(current_move_str)
            if is_black_turn:
                current_move = mirror_move(
                    current_move
                )  # Unflip move to be from white's perspective, rather than the current player's perspective
            preds_list.append((current_move, probs[legal_move_index].item()))
        best_move = preds_list[probs[legal_move_indices].argmax().item()][0]
        preds_list.sort(key=lambda x: x[1], reverse=True)

        best_move_san = leela_board.board.san(best_move)
        best_move_uci = str(best_move)

        if is_black_turn:
            win_probability = 1 - logits_value.item()
        else:
            win_probability = logits_value.item()
        return Maia2ChessMovePrediction(
            leela_board=leela_board,
            best_move_san=best_move_san,
            best_move_uci=best_move_uci,
            preds_list=dict(preds_list),
            win_probability=win_probability,
        )

    @torch.inference_mode(True)
    def make_prediction(
        self,
        fens: str | list[str],
        elo_self: int | list[int],
        elo_oppo: int | list[int],
        *args,
        batch_size: int = 16,
        tqdm_leave: bool = False,
        persistent_workers: bool = False,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        """Make a prediction based on the provided FEN string.

        Returns a dataclass of the board, the best move in SAN format, the best move in UCI format and a dictionary
        of all legal moves with their probabilities.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.input_config.validate(
            fens=fens,
            elo_self=elo_self,
            elo_opponent=elo_oppo,
            **extract_domain_inputs(kwargs),
        )
        fen_count = 1 if isinstance(fens, str) else len(fens)
        resolved_workers = resolve_num_workers(
            fen_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )
        dataset = Maia2Dataset(
            fens=fens,
            elo_self=elo_self,
            elo_oppo=elo_oppo,
        )
        is_using_cuda = torch.cuda.is_available() and self.model.device.type == "cuda"

        raw_outputs = []
        for (
            board_tensor,
            elo_self_tensor,
            elo_oppo_tensor,
            legal_moves_mask,
            lboard,
        ) in progress(
            iter_dataset_batches(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                num_workers=resolved_workers,
                pin_memory=is_using_cuda,
                persistent_workers=persistent_workers,
            ),
            desc="Making predictions",
            total=(len(dataset) + batch_size - 1) // batch_size,
            leave=tqdm_leave,
        ):
            probs, logits_value = self.forward(
                board_tensor,
                elo_self_tensor,
                elo_oppo_tensor,
                legal_moves_mask,
            )
            raw_outputs.extend(
                zip(
                    lboard,
                    legal_moves_mask,
                    probs,
                    logits_value,
                )
            )
        if raw_outputs:
            del board_tensor, elo_self_tensor, elo_oppo_tensor
            predictions = adaptive_process_map(
                self.post_process_predictions,
                *zip(*raw_outputs),
                parallel_threshold=parallel_threshold,
                max_workers=resolved_workers,
                desc="Collecting predictions",
                leave=tqdm_leave,
                chunksize=64,
            )
        else:
            predictions = []
        return ChessMovePredictionDataset[Maia2ChessMovePrediction](
            predictions=predictions
        )


__all__ = [
    "Maia2ChessMovePrediction",
    "Maia2Dataset",
    "Maia2Pipeline",
]
