"""Allie inference pipeline."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import pathlib
import typing as tp

import chess
import chess.pgn
import torch
import torch.nn.functional as F
import torch.utils.data

from chessnets.models.allie import configuration_allie as allie_config
from chessnets.models.allie import data as allie_data
from chessnets.models.allie import modeling_allie as allie_model
from chessnets.models.allie import moves as allie_moves
from chessnets.models.allie import utils as allie_utils

from ..base import Pipeline, PipelineInputConfig, extract_domain_inputs
from ..outputs import AllieMovePrediction, ChessMovePredictionDataset
from ..pipeline_utils import (
    progress,
    DEFAULT_PARALLEL_THRESHOLD,
    adaptive_process_map,
    iter_dataset_batches,
    resolve_num_workers,
)

ALLIE_PRECISIONS = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class AllieDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: allie_data.UCITokenizer,
        pgns: str | list[str],
        time_controls,
        white_elos,
        black_elos,
        moves_seconds: list[int] | list[list[int]] | list[None] | None = None,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
        if isinstance(time_controls, str):
            time_controls = [time_controls]
        if isinstance(white_elos, int):
            white_elos = [white_elos]
        if isinstance(black_elos, int):
            black_elos = [black_elos]
        assert len(pgns) == len(time_controls) == len(white_elos) == len(black_elos), (
            "Length of inputs must match."
        )

        if moves_seconds is None:
            moves_seconds = [None] * len(pgns)
        elif isinstance(moves_seconds, list) and all(
            isinstance(ms, int) for ms in moves_seconds
        ):
            moves_seconds = [[moves_seconds]]
        assert len(moves_seconds) == len(pgns), (
            "Length of moves_seconds must match PGNs."
        )
        self.tokenizer = tokenizer
        self.pgn = pgns
        self.time_control = time_controls
        self.white_elo = white_elos
        self.black_elo = black_elos
        self.moves_seconds = moves_seconds

    def __len__(self):
        return len(self.pgn)

    def __getitem__(self, idx):
        pgn = self.pgn[idx]
        time_control = self.time_control[idx]
        white_elo = self.white_elo[idx]
        black_elo = self.black_elo[idx]
        moves_seconds = self.moves_seconds[idx]

        chess_game = chess.pgn.read_game(io.StringIO(pgn))
        if chess_game is None:
            raise ValueError(f"Invalid PGN: {pgn}")
        moves = [move.uci() for move in chess_game.mainline_moves()]
        if moves_seconds is None:
            moves_seconds = [None] * len(moves)
        moves_seconds_ = []
        for node, s in zip(chess_game.mainline(), moves_seconds):
            moves_seconds_.append(node.clock() or s or 0)

        return allie_data.Game(
            time_control=time_control,
            white_elo=white_elo,
            black_elo=black_elo,
            outcome=None,
            normal_termination=False,
            moves=moves,
            moves_seconds=moves_seconds_,
            next_move_seconds=None,
        )

    def collate_fn(
        self, batch: list[allie_data.Game]
    ) -> tuple[
        list[allie_data.Game],
        allie_data.Batch,
        list[list[str]],
        torch.Tensor,
    ]:
        legal_moves = [game.legal_moves for game in batch]
        token_batch = self.tokenizer.pad_and_collate(list(batch))
        pov = torch.tensor(
            [1 if game.board.turn == chess.WHITE else -1 for game in batch],
            dtype=torch.int8,
        )
        return batch, token_batch, legal_moves, pov


class AlliePipeline(Pipeline):
    def __init__(
        self,
        model: allie_model.CausalLMWithRegressionHead,
        tokenizer: allie_data.UCITokenizer,
        precision: tp.Literal["float16", "bfloat16", "float32"] = "float32",
        temperature: float = 1.0,
        time_prediction: bool = True,
        input_config: PipelineInputConfig | None = None,
    ):
        super().__init__(input_config=input_config)
        self.model = model
        self.tokenizer = tokenizer
        self.device = str(self.model.device)
        self.ctx = (
            contextlib.nullcontext()
            if self.device == "cpu"
            else torch.amp.autocast(
                device_type="cuda", dtype=ALLIE_PRECISIONS[precision]
            )
        )
        self.temperature = max(1e-9, temperature)
        self.time_prediction = time_prediction

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(
            required=("pgns", "time_controls", "white_elos", "black_elos"),
            optional=("move_seconds",),
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path = (
            "yimingzhang/allie-models"
        ),
        device: str | torch.device = "cpu",
        *,
        variant: str = "medium",
        revision: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        repo_type: str | None = "dataset",
        **kwargs,
    ) -> tp.Self:
        # Before the common loader API, ``device`` was the first positional
        # argument.  Continue accepting that form while model IDs become the
        # canonical first argument.
        if isinstance(pretrained_model_name_or_path, torch.device):
            device = pretrained_model_name_or_path
            pretrained_model_name_or_path = "yimingzhang/allie-models"
        elif isinstance(pretrained_model_name_or_path, str) and (
            pretrained_model_name_or_path in {"cpu", "cuda", "mps"}
            or pretrained_model_name_or_path.startswith("cuda:")
        ):
            device = pretrained_model_name_or_path
            pretrained_model_name_or_path = "yimingzhang/allie-models"
        config = allie_config.AllieConfig.from_pretrained(
            pretrained_model_name_or_path,
            variant=variant,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            repo_type=repo_type,
        )
        tokenizer = allie_data.UCITokenizer(
            **dataclasses.asdict(config.data_config.tokenizer_config)
        )
        model = allie_model.CausalLMWithRegressionHead.from_pretrained(
            pretrained_model_name_or_path,
            variant=variant,
            tokenizer=tokenizer,
            config=config,
            device=device,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            repo_type=repo_type,
        )
        return cls(model, tokenizer, **kwargs)

    @torch.inference_mode(True)
    def forward(
        self,
        token_batch: allie_data.Batch,
        pov: torch.Tensor,
        vocab_size: int,
    ):
        token_batch = allie_utils.to_device(token_batch, self.device)

        with self.ctx:
            outputs = self.model(**token_batch)
            last_index = token_batch["position_ids"][:, -1].unsqueeze(-1)
            last_index_logits = last_index.unsqueeze(-1).expand(-1, -1, vocab_size)
        logits = (
            torch.gather(outputs["logits"], dim=1, index=last_index_logits)
            .squeeze(1)
            .float()
            .cpu()
        )  # (batch_size, sequence_length, vocab_size) -> (batch_size, vocab_size)

        if self.time_prediction:
            time_encoding = (
                torch.gather(outputs["time_logits"], dim=1, index=last_index)
                .squeeze(1)
                .cpu()
            )  # (batch_size, sequence_length) -> (batch_size,)
            think_time = allie_data.undo_time_normalization(time_encoding)
        else:
            think_time = [None] * logits.shape[0]
        value_logits = (
            torch.gather(outputs["value_logits"], dim=1, index=last_index)
            .squeeze(1)
            .cpu()
        )  # (batch_size, sequence_length) -> (batch_size,)

        value = pov * value_logits

        resigned_logit = logits[:, self.tokenizer.resigned_token_id]

        all_moves_logits = logits[:, : allie_moves.NUM_CHESS_MOVES]
        all_moves_probabilities = F.softmax(all_moves_logits / self.temperature, dim=1)

        top_move_idx = all_moves_probabilities.argmax(dim=1)

        return (
            logits,
            resigned_logit,
            value_logits,
            value,
            top_move_idx,
            all_moves_probabilities,
            think_time,
        )

    @staticmethod
    def post_process_predictions(
        tokenizer,
        temperature,
        game,
        legal_moves,
        logits,
        resigned_logit,
        value_logits,
        value,
        top_move_idx,
        all_move_probabilities,
        think_time,
    ) -> AllieMovePrediction:
        legal_moves_tensor = torch.tensor(
            [tokenizer.token_to_id[move] for move in legal_moves],
            dtype=torch.int64,
        )

        move_logits = logits[legal_moves_tensor]
        r = resigned_logit > move_logits.max() and value < -0.9

        is_top_move_legal = tokenizer.tokens[top_move_idx] in legal_moves
        legal_move_probability = all_move_probabilities[legal_moves_tensor].sum().item()

        best_move_uci = legal_moves[move_logits.argmax().item()]
        best_move = chess.Move.from_uci(best_move_uci)
        best_move_san = game.board.san(best_move)
        preds_legal = F.softmax(move_logits / temperature, dim=0)
        preds_list_legal = [
            (chess.Move.from_uci(m), p.item()) for m, p in zip(legal_moves, preds_legal)
        ]
        preds_list_legal.sort(key=lambda x: x[1], reverse=True)
        preds_list = [
            (chess.Move.from_uci(m), p.item())
            for m, p in zip(
                allie_moves.CHESS_MOVES,
                all_move_probabilities,
            )
        ]
        preds_list.sort(key=lambda x: x[1], reverse=True)
        game.make_move(best_move_uci)
        win_probability = (value_logits.item() + 1) / 2

        return AllieMovePrediction(
            game=game,
            best_move_san=best_move_san,
            best_move_uci=best_move_uci,
            preds_list_legal=dict(preds_list_legal),
            preds_list=dict(preds_list),
            resigned=r.item(),
            win_probability=win_probability,
            think_time=think_time.item() if think_time is not None else None,
            is_top_move_legal=is_top_move_legal,
            legal_move_probability=legal_move_probability,
        )

    def make_prediction(
        self,
        pgn: str | list[str],
        time_control: str | list[str],
        white_elo: int | list[int],
        black_elo: int | list[int],
        moves_seconds: list[int] | list[list[int]] | list[None] | None = None,
        *args,
        batch_size: int = 16,
        tqdm_leave: bool = False,
        persistent_workers: bool = False,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.input_config.validate(
            pgn=pgn,
            time_control=time_control,
            white_elo=white_elo,
            black_elo=black_elo,
            move_seconds=moves_seconds,
            **extract_domain_inputs(kwargs),
        )
        pgn_count = 1 if isinstance(pgn, str) else len(pgn)
        resolved_workers = resolve_num_workers(
            pgn_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )
        dataset = AllieDataset(
            tokenizer=self.tokenizer,
            pgns=pgn,
            time_controls=time_control,
            white_elos=white_elo,
            black_elos=black_elo,
            moves_seconds=moves_seconds,
        )
        is_using_cuda = torch.cuda.is_available() and self.model.device.type == "cuda"

        raw_outputs = []
        for games, token_batch, legal_moves, pov in progress(
            iter_dataset_batches(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                num_workers=resolved_workers,
                pin_memory=is_using_cuda,
                persistent_workers=persistent_workers,
            ),
            desc="Preparing games",
            total=(len(dataset) + batch_size - 1) // batch_size,
            leave=tqdm_leave,
        ):
            (
                logits,
                resigned_logit,
                value_logits,
                value,
                top_move_idx,
                all_moves_probabilities,
                think_time,
            ) = self.forward(token_batch, pov, len(self.tokenizer))
            for game, lm, lg, rl, vl, v, tmi, amp, tt in zip(
                games,
                legal_moves,
                logits,
                resigned_logit,
                value_logits,
                value,
                top_move_idx,
                all_moves_probabilities,
                think_time,
            ):
                raw_outputs.append(
                    (
                        self.tokenizer,
                        self.temperature,
                        game,
                        lm,
                        lg,
                        rl,
                        vl,
                        v,
                        tmi,
                        amp,
                        tt,
                    )
                )
        if raw_outputs:
            del token_batch
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
        return ChessMovePredictionDataset[AllieMovePrediction](predictions=predictions)


__all__ = [
    "ALLIE_PRECISIONS",
    "AllieDataset",
    "AllieMovePrediction",
    "AlliePipeline",
]
