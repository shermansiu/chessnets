"""ChessCLIP text/image scoring pipeline."""

from __future__ import annotations

import typing as tp

import chess
import torch
import torch.utils.data

from chessnets._optional import import_optional
from chessnets.utils import board_history

from ..base import Pipeline, PipelineInputConfig, extract_domain_inputs
from ..outputs import ChessMovePrediction, ChessMovePredictionDataset
from ..pipeline_utils import (
    progress,
    DEFAULT_PARALLEL_THRESHOLD,
    adaptive_process_map,
    iter_dataset_batches,
    resolve_num_workers,
)


class OpenClipTokenizer(tp.Protocol):
    def __call__(
        self, texts: str | list[str], context_length: int = 77
    ) -> torch.LongTensor: ...


class ChessClipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pgns: list[str] | str,
        prompts: list[str] | str,
        tokenizer: OpenClipTokenizer,
        *,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
        if isinstance(prompts, str):
            prompts = [prompts]
        assert len(pgns) == len(prompts), "PGNs and prompts must have the same length."
        self.histories = adaptive_process_map(
            board_history.BoardHistory.from_pgn,
            pgns,
            parallel_threshold=parallel_threshold,
            max_workers=num_workers,
            chunksize=16,
            desc="Converting PGNs to board histories",
            leave=False,
        )
        self.prompts = prompts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    @staticmethod
    def load_board_history_info(history: board_history.BoardHistory):
        boards = []
        actions = []
        legal_moves = list(history.active_board.legal_moves)
        for legal_move in legal_moves:
            history_with_new_move = history.copy_with_move(legal_move)
            boards.append(history_with_new_move.asarray())
            actions.append(history_with_new_move.move_tensor_target())
        board_tensor = torch.stack([torch.from_numpy(b) for b in boards]).reshape(
            -1, 112, 8, 8
        )
        action_tensor = torch.stack([torch.from_numpy(a) for a in actions])
        return board_tensor, action_tensor, legal_moves

    def __getitem__(self, idx):
        history = self.histories[idx]
        prompt = self.prompts[idx]
        board_tensor, action_tensor, legal_moves = self.load_board_history_info(history)
        seq_length = len(legal_moves)
        return board_tensor, action_tensor, prompt, history, legal_moves, seq_length

    def collate_fn(self, batch):
        (
            board_tensors,
            action_tensors,
            prompts,
            histories,
            all_legal_moves,
            seq_lengths,
        ) = zip(*batch)
        board_tensor = torch.cat(board_tensors, dim=0)
        action_tensor = torch.cat(action_tensors, dim=0)
        text_tokens = self.tokenizer(list(prompts))
        return (
            board_tensor,
            action_tensor,
            text_tokens,
            list(histories),
            list(all_legal_moves),
            list(seq_lengths),
        )


class ChessClipPipeline(Pipeline):
    auto_batch_sizes = {"cpu": 64, "cuda": 256}

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: OpenClipTokenizer,
        temperature: float = 1.0,
        input_config: PipelineInputConfig | None = None,
    ):
        super().__init__(input_config=input_config)
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.temperature = temperature

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(required=("pgns", "prompts"))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "Waterhorse/ChessCLIP",
        device: tp.Any = "cpu",
        **kwargs: tp.Any,
    ) -> tp.Self:
        """Load the published ChessCLIP checkpoint."""
        if pretrained_model_name_or_path != "Waterhorse/ChessCLIP":
            raise ValueError(
                "ChessClipPipeline currently supports only the published "
                "'Waterhorse/ChessCLIP' checkpoint."
            )
        open_clip = import_optional("open_clip", "chess-clip")
        from chessnets.models import chess_clip

        model_name = kwargs.pop("model_name", "chessclip-quickgelu")
        tokenizer = open_clip.get_tokenizer(model_name)
        model_kwargs = dict(kwargs.pop("model_kwargs", {}))
        model_kwargs["model_name"] = model_name
        model_kwargs.setdefault("device", device)
        model = chess_clip.ChessCLIP.from_pretrained(
            pretrained_model_name_or_path, **model_kwargs
        )
        model.eval()
        return cls(model=model, tokenizer=tokenizer, **kwargs)

    @torch.inference_mode(True)
    def forward(
        self, board_tensor, action_tensor, text_tokens, seq_lengths
    ) -> torch.Tensor:
        assert isinstance(self.model.encode_image, tp.Callable)
        assert isinstance(self.model.encode_text, tp.Callable)
        image_features = self.model.encode_image(
            (
                board_tensor.to(
                    device=self.model.device,
                    dtype=self.model.dtype,
                    non_blocking=board_tensor.is_pinned(),
                ),
                action_tensor.to(
                    device=self.model.device,
                    dtype=self.model.dtype,
                    non_blocking=action_tensor.is_pinned(),
                ),
            )
        ).float()
        text_features = self.model.encode_text(
            text_tokens.to(device=self.model.device)
        ).float()

        image_features /= image_features.norm(
            dim=-1, keepdim=True
        )  # Norm for cosine similarity
        text_features /= text_features.norm(
            dim=-1, keepdim=True
        )  # (batch_size, embed_dim)
        image_features = torch.nested.nested_tensor(
            list(torch.split(image_features, seq_lengths, dim=0)), layout=torch.jagged
        )  # (batch_size, jagged_action_dim, embed_dim)
        # According to the README, you need a temperature of 0.01 (i.e. x by 100) to get peakier probabilities.
        # https://github.com/mlfoundations/open_clip
        similarity = (
            (image_features / self.temperature) @ text_features.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, jagged_action_dim)
        similarity = similarity.softmax(dim=1)  # Convert similarities to probabilities
        return similarity.cpu()

    @staticmethod
    def post_process_predictions(
        similarity,
        history: board_history.BoardHistory,
        legal_moves: list[chess.Move],
        return_histories: bool = False,
    ):
        best_move = legal_moves[similarity.argmax().item()]
        best_move_san = history.active_board.san(best_move)
        best_move_uci = str(best_move)
        preds_list = {m: similarity[i].item() for i, m in enumerate(legal_moves)}
        if return_histories:
            history.append(best_move)
        return ChessMovePrediction(
            board_history=history if return_histories else None,
            best_move_san=best_move_san,
            best_move_uci=best_move_uci,
            preds_list=preds_list,
        )

    def make_prediction(
        self,
        pgns: str | list[str],
        prompts: str | list[str],
        return_histories: bool = False,
        tqdm_leave: bool = False,
        batch_size: int = 16,
        persistent_workers: bool = False,
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.input_config.validate(
            pgns=pgns, prompts=prompts, **extract_domain_inputs(kwargs)
        )
        pgn_count = 1 if isinstance(pgns, str) else len(pgns)
        resolved_workers = resolve_num_workers(
            pgn_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )
        dataset = ChessClipDataset(
            pgns=pgns,
            prompts=prompts,
            tokenizer=self.tokenizer,
            num_workers=resolved_workers,
            parallel_threshold=parallel_threshold,
        )
        is_using_cuda = torch.cuda.is_available() and self.model.device.type == "cuda"

        raw_outputs = []
        for (
            board_tensor,
            action_tensor,
            text_tokens,
            histories,
            all_legal_moves,
            seq_lengths,
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
            similarity = self.forward(
                board_tensor, action_tensor, text_tokens, seq_lengths
            )
            del board_tensor, action_tensor, text_tokens
            raw_outputs.extend(
                zip(
                    similarity.unbind(),
                    histories,
                    all_legal_moves,
                    [return_histories] * len(histories),
                )
            )
        if raw_outputs:
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
        return ChessMovePredictionDataset[ChessMovePrediction](predictions=predictions)


__all__ = [
    "ChessClipDataset",
    "ChessClipPipeline",
    "OpenClipTokenizer",
]
