"""UniMaia inference pipeline with explicit input contracts."""

from __future__ import annotations

import dataclasses
import enum
import pathlib
import typing as tp

import torch
import torch.nn.functional as F
import torch.utils.data

from chessnets.models.unimaia import modeling_unimaia as unimaia_model
from chessnets.models.unimaia.configuration_unimaia import UNIMAIA_OUTCOME_LABELS
from chessnets.utils import board_history
from chessnets.utils.time_control import (
    TimeInfoPromptStyle,
    append_basic_time_info_to_prompt,
    time_control_to_time_control_metadata,
)

from ..base import PipelineInputConfig, extract_domain_inputs
from ..lc0.pipeline_lc0 import Lc0Pipeline
from ..outputs import (
    AuxiliaryTargetChessMovePrediction,
    ChessMovePredictionDataset,
)
from ..pipeline_utils import (
    DEFAULT_PARALLEL_THRESHOLD,
    adaptive_process_map,
    iter_dataset_batches,
    progress,
    resolve_num_workers,
)

END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE = "{prompt}<|endoftext|>{pgn}"
all_termination_reasons = UNIMAIA_OUTCOME_LABELS


class UniMaiaInputMode(str, enum.Enum):
    PROMPT = "prompt"
    TIME_CONTROL = "time_control"
    CLOCKS = "clocks"


@dataclasses.dataclass(frozen=True)
class UniMaiaPipelineConfig:
    """Select the input contract used by :class:`UniMaiaPipeline`."""

    input_mode: UniMaiaInputMode = UniMaiaInputMode.PROMPT
    time_info_prompt_style: TimeInfoPromptStyle = TimeInfoPromptStyle.REWRITTEN

    def __post_init__(self):
        if not isinstance(self.input_mode, UniMaiaInputMode):
            try:
                object.__setattr__(
                    self, "input_mode", UniMaiaInputMode(self.input_mode)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported UniMaia input mode: {self.input_mode!r}"
                ) from exc
        if not isinstance(self.time_info_prompt_style, TimeInfoPromptStyle):
            try:
                object.__setattr__(
                    self,
                    "time_info_prompt_style",
                    TimeInfoPromptStyle(self.time_info_prompt_style),
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported UniMaia time-info prompt style: "
                    f"{self.time_info_prompt_style!r}"
                ) from exc

    @property
    def input_config(self) -> PipelineInputConfig:
        if self.input_mode == UniMaiaInputMode.PROMPT:
            return PipelineInputConfig.from_inputs(required=("pgns", "prompts"))
        if self.input_mode == UniMaiaInputMode.TIME_CONTROL:
            return PipelineInputConfig.from_inputs(
                required=("pgns", "prompts", "time_controls")
            )
        return PipelineInputConfig.from_inputs(
            required=(
                "pgns",
                "prompts",
                "time_controls",
                "white_clock_times",
                "black_clock_times",
            ),
            paired=(("white_clock_times", "black_clock_times"),),
        )


class UniMaiaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pgns: list[str] | str,
        prompts: list[str] | str,
        should_add_endoftext_token_and_pgn: bool = False,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
        if isinstance(prompts, str):
            prompts = [prompts]
        self.pgns = pgns
        self.prompts = prompts
        self.should_add_endoftext_token_and_pgn = should_add_endoftext_token_and_pgn
        if len(pgns) != len(prompts):
            raise ValueError("PGNs and prompts must have the same length")

    def __len__(self):
        return len(self.pgns)

    def __getitem__(self, idx):
        pgn = self.pgns[idx]
        history = board_history.BoardHistory.from_pgn(pgn)
        board_array = history.asarray()
        prompt = self.prompts[idx]
        if self.should_add_endoftext_token_and_pgn:
            prompt = END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE.format(
                prompt=prompt, pgn=pgn
            )
        return (
            torch.from_numpy(board_array).reshape(
                112,
                64,  # Reshape to (channels, height * width)
            ),
            prompt,
            history,
        )

    @staticmethod
    def collate_fn(batch):
        board_tensors, prompts, histories = zip(*batch)
        board_tensor = torch.stack(board_tensors)
        return board_tensor, list(prompts), list(histories)


class UniMaiaDatasetWithTimeControl(UniMaiaDataset):
    def __init__(
        self,
        pgns: list[str] | str,
        prompts: list[str] | str,
        time_controls: list[str] | str,
        should_add_endoftext_token_and_pgn: bool = False,
        time_info_prompt_style: TimeInfoPromptStyle = TimeInfoPromptStyle.REWRITTEN,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(time_controls, str):
            time_controls = [time_controls]
        self.pgns = pgns
        self.prompts = prompts
        self.time_controls = time_controls
        self.should_add_endoftext_token_and_pgn = should_add_endoftext_token_and_pgn
        self.time_info_prompt_style = TimeInfoPromptStyle(time_info_prompt_style)
        if not (len(pgns) == len(prompts) == len(time_controls)):
            raise ValueError(
                "PGNs, prompts, and time controls must have the same length"
            )

    def __getitem__(self, idx: int):
        pgn = self.pgns[idx]
        history = board_history.BoardHistory.from_pgn(pgn)
        board_array = history.asarray()
        prompt = self.prompts[idx]
        time_control = self.time_controls[idx]
        time_control_metadata = time_control_to_time_control_metadata(time_control)
        turn_index = len(history.moves)
        if self.should_add_endoftext_token_and_pgn:
            prompt = END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE.format(
                prompt=prompt, pgn=pgn
            )
        prompt = append_basic_time_info_to_prompt(
            history,
            turn_index,
            time_control_metadata,
            prompt,
            self.time_info_prompt_style,
        )
        return (
            torch.from_numpy(board_array).reshape(
                112,
                64,  # Reshape to (channels, height * width)
            ),
            prompt,
            history,
        )


class UniMaiaDatasetWithClockInfo(UniMaiaDataset):
    @staticmethod
    def _as_per_game_clock_times(
        values: list[float | None] | list[list[float | None]], name: str
    ) -> list[list[float | None]]:
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"{name} must be a list of clock histories")
        if not values or not isinstance(values[0], (list, tuple)):
            return [list(values)]
        return [list(history) for history in values]

    def __init__(
        self,
        pgns: list[str] | str,
        prompts: list[str] | str,
        time_controls: list[str] | str,
        white_clock_times: list[list[float | None]] | list[float | None],
        black_clock_times: list[list[float | None]] | list[float | None],
        should_add_endoftext_token_and_pgn: bool = False,
        time_info_prompt_style: TimeInfoPromptStyle = TimeInfoPromptStyle.REWRITTEN,
    ):
        if isinstance(pgns, str):
            pgns = [pgns]
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(time_controls, str):
            time_controls = [time_controls]
        white_clock_times = self._as_per_game_clock_times(
            white_clock_times, "white_clock_times"
        )
        black_clock_times = self._as_per_game_clock_times(
            black_clock_times, "black_clock_times"
        )
        self.pgns = pgns
        self.prompts = prompts
        self.time_controls = time_controls
        self.white_clock_times: list[list[float | None]] = white_clock_times
        self.black_clock_times: list[list[float | None]] = black_clock_times
        self.should_add_endoftext_token_and_pgn = should_add_endoftext_token_and_pgn
        self.time_info_prompt_style = TimeInfoPromptStyle(time_info_prompt_style)
        if not (
            len(pgns)
            == len(prompts)
            == len(time_controls)
            == len(white_clock_times)
            == len(black_clock_times)
        ):
            raise ValueError(
                "PGNs, prompts, time controls, and clock times must have the same length"
            )

    def __getitem__(self, idx: int):
        pgn = self.pgns[idx]
        history = board_history.BoardHistory.from_pgn(pgn)
        white_clock_times = self.white_clock_times[idx]
        black_clock_times = self.black_clock_times[idx]
        time_control = self.time_controls[idx]
        history.time_control = time_control
        history.white_clock_times = white_clock_times
        history.black_clock_times = black_clock_times
        board_array = history.asarray()
        prompt = self.prompts[idx]
        time_control_metadata = time_control_to_time_control_metadata(time_control)
        turn_index = len(history.moves)
        if self.should_add_endoftext_token_and_pgn:
            prompt = END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE.format(
                prompt=prompt, pgn=pgn
            )
        prompt = append_basic_time_info_to_prompt(
            history,
            turn_index,
            time_control_metadata,
            prompt,
            self.time_info_prompt_style,
        )
        return (
            torch.from_numpy(board_array).reshape(
                112,
                64,  # Reshape to (channels, height * width)
            ),
            prompt,
            history,
        )


all_termination_reasons = UNIMAIA_OUTCOME_LABELS


class UniMaiaPipeline(Lc0Pipeline):
    # UniMaia runs a language model next to the Lc0 network.
    auto_batch_sizes = {"cpu": 8, "cuda": 8}

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: tp.Any,
        temperature: float = 1.0,
        should_merge_lora_adapter: bool = False,
        max_seq_len: int = 1024,
        should_add_endoftext_token_and_pgn: bool = False,
        pipeline_config: UniMaiaPipelineConfig | None = None,
        input_config: PipelineInputConfig | None = None,
    ):
        self.pipeline_config = pipeline_config or UniMaiaPipelineConfig()
        expected_input_config = self.pipeline_config.input_config
        if input_config is not None:
            if input_config.required_inputs != expected_input_config.required_inputs:
                raise ValueError(
                    "The supplied input_config is incompatible with the selected "
                    f"UniMaia input mode ({self.pipeline_config.input_mode.value})"
                )
            if (
                input_config.allowed_inputs is not None
                and input_config.allowed_inputs != expected_input_config.allowed_inputs
            ):
                raise ValueError(
                    "The supplied input_config accepts inputs incompatible with "
                    f"UniMaia input mode ({self.pipeline_config.input_mode.value})"
                )
        input_config = expected_input_config
        super().__init__(model, temperature=temperature, input_config=input_config)
        # Merge in LoRA weights for slightly faster inference
        if (
            should_merge_lora_adapter
            and hasattr(self.model, "text_encoder")
            and hasattr(self.model.text_encoder, "merge_and_unload")
        ):
            self.model.text_encoder = self.model.text_encoder.merge_and_unload()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.should_add_endoftext_token_and_pgn = should_add_endoftext_token_and_pgn
        self.should_output_value = False
        self.should_output_moves_left = False
        if isinstance(model, unimaia_model.UniMaiaModel):
            self.should_output_value = model.umt_config.should_use_value_loss
            self.should_output_moves_left = model.umt_config.should_use_moves_left_loss

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(required=("pgns", "prompts"))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | pathlib.Path,
        device: tp.Any = "cpu",
        dtype: tp.Any = None,
        should_merge_lora_adapter: bool = False,
        pipeline_config: UniMaiaPipelineConfig | None = None,
        config: tp.Any = None,
        checkpoint_format: tp.Literal["safetensors", "training_pt"] = "safetensors",
        revision: str | None = None,
        cache_dir: str | pathlib.Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        repo_type: str | None = None,
        **kwargs: tp.Any,
    ) -> tp.Self:
        """Load a UniMaia checkpoint and its tokenizer."""
        from chessnets.loaders import load_unimaia_model

        model, tokenizer = load_unimaia_model(
            pretrained_model_name_or_path,
            is_eval=True,
            device=device,
            dtype=dtype,
            should_merge_lora_adapter=should_merge_lora_adapter,
            config=config,
            checkpoint_format=checkpoint_format,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            repo_type=repo_type,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            should_merge_lora_adapter=should_merge_lora_adapter,
            pipeline_config=pipeline_config,
            **kwargs,
        )

    @torch.inference_mode(True)
    def forward(
        self,
        board_tensor: torch.Tensor,
        prompts: list[str],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        assert len(prompts) == board_tensor.shape[0], (
            f"Text tokens batch size ({len(prompts)}) must match batch size ({board_tensor.shape[0]})."
        )

        text_tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        out = self.model(
            board_tensor.to(
                device=self.model.device,
                dtype=self.model.dtype,
                non_blocking=board_tensor.is_pinned(),
            ),
            text_tokens.to(self.model.device),
        )
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

        if hasattr(out, "moves_left") and self.should_output_moves_left:
            moves_left = out.moves_left.cpu()
        else:
            moves_left = None

        if hasattr(out, "outcome") and out.outcome is not None:
            outcome = F.softmax(out.outcome, dim=1).cpu()
        else:
            outcome = None

        if hasattr(out, "move_delay") and out.move_delay is not None:
            move_delay = out.move_delay.cpu()
        else:
            move_delay = None

        del board_tensor, text_tokens, out  # Free memory
        return policy, result, value, moves_left, outcome, move_delay

    @staticmethod
    def post_process_predictions(
        policy,
        result,
        value,
        moves_left,
        outcome,
        move_delay,
        history: board_history.BoardHistory,
        temperature: float,
        return_histories: bool = False,
    ):
        old_pred = Lc0Pipeline.post_process_predictions(
            policy, result, value, moves_left, history, temperature, return_histories
        )
        prediction = AuxiliaryTargetChessMovePrediction(
            old_pred.board_history,
            old_pred.best_move_san,
            old_pred.best_move_uci,
            old_pred.preds_list,
            win_probability=old_pred.win_probability,
            result_probs=old_pred.result_probs,
            result=old_pred.result,
            moves_left=old_pred.moves_left,
        )
        if outcome is not None:
            prediction.outcome_type_probs = {
                reason: outcome[i].item()
                for i, reason in enumerate(all_termination_reasons)
            }
            prediction.outcome_type = all_termination_reasons[outcome.argmax().item()]
        if move_delay is not None:
            prediction.move_delay = move_delay.item()
        return prediction

    def make_prediction(
        self,
        pgns: str | list[str],
        prompts: str | list[str],
        time_controls: str | list[str] | None = None,
        white_clock_times: list[float | None] | list[list[float | None]] | None = None,
        black_clock_times: list[float | None] | list[list[float | None]] | None = None,
        max_seq_len: int = 1024,
        batch_size: int = 16,
        return_histories: bool = False,
        tqdm_leave: bool = False,
        shuffle: bool = False,
        persistent_workers: bool = False,  # May cause an open file descriptor leak if True, though it can prevent code from crashing if repeatedly making inference calls in a loop
        num_workers: int | None = None,
        parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD,
        **kwargs,
    ) -> ChessMovePredictionDataset:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.max_seq_len = max_seq_len
        self.input_config.validate(
            pgns=pgns,
            prompts=prompts,
            time_controls=time_controls,
            white_clock_times=white_clock_times,
            black_clock_times=black_clock_times,
            **extract_domain_inputs(kwargs),
        )
        pgn_count = 1 if isinstance(pgns, str) else len(pgns)
        resolved_workers = resolve_num_workers(
            pgn_count,
            requested_workers=num_workers,
            parallel_threshold=parallel_threshold,
        )
        optional_inputs = {
            "time_controls": time_controls,
            "clock_times": (
                (white_clock_times, black_clock_times)
                if white_clock_times is not None or black_clock_times is not None
                else None
            ),
        }
        mode = self.pipeline_config.input_mode
        if mode == UniMaiaInputMode.PROMPT and any(
            value is not None for value in optional_inputs.values()
        ):
            raise ValueError(
                "Prompt-only UniMaia inference does not accept time-control or clock inputs"
            )
        if mode == UniMaiaInputMode.TIME_CONTROL and (
            white_clock_times is not None or black_clock_times is not None
        ):
            raise ValueError(
                "Time-control UniMaia inference does not accept clock histories"
            )
        if mode == UniMaiaInputMode.PROMPT:
            dataset = UniMaiaDataset(
                pgns=pgns,
                prompts=prompts,
                should_add_endoftext_token_and_pgn=self.should_add_endoftext_token_and_pgn,
            )
        elif mode == UniMaiaInputMode.TIME_CONTROL:
            self.input_config.validate(
                pgns=pgns, prompts=prompts, time_controls=time_controls
            )
            dataset = UniMaiaDatasetWithTimeControl(
                pgns=pgns,
                prompts=prompts,
                time_controls=time_controls,
                should_add_endoftext_token_and_pgn=self.should_add_endoftext_token_and_pgn,
                time_info_prompt_style=self.pipeline_config.time_info_prompt_style,
            )
        else:
            self.input_config.validate(
                pgns=pgns,
                prompts=prompts,
                time_controls=time_controls,
                white_clock_times=white_clock_times,
                black_clock_times=black_clock_times,
            )
            if white_clock_times is None or black_clock_times is None:
                raise ValueError(
                    "white_clock_times and black_clock_times are required for clock-aware input"
                )
            dataset = UniMaiaDatasetWithClockInfo(
                pgns=pgns,
                prompts=prompts,
                time_controls=time_controls,
                white_clock_times=white_clock_times,
                black_clock_times=black_clock_times,
                should_add_endoftext_token_and_pgn=self.should_add_endoftext_token_and_pgn,
                time_info_prompt_style=self.pipeline_config.time_info_prompt_style,
            )
        is_using_cuda = torch.cuda.is_available() and self.model.device.type == "cuda"
        raw_outputs = []
        for board_tensor, prompts_, histories in progress(
            iter_dataset_batches(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=dataset.collate_fn,
                num_workers=resolved_workers,
                pin_memory=is_using_cuda,
                persistent_workers=persistent_workers,
            ),
            desc="Making predictions",
            total=(len(dataset) + batch_size - 1) // batch_size,
            leave=tqdm_leave,
        ):
            policy, result, value, moves_left, outcome, move_delay = self.forward(
                board_tensor, prompts_
            )
            if result is None:
                result = [None] * len(histories)
            if value is None:
                value = [None] * len(histories)
            if moves_left is None:
                moves_left = [None] * len(histories)
            if outcome is None:
                outcome = [None] * len(histories)
            if move_delay is None:
                move_delay = [None] * len(histories)
            for p, r, v, ml, o, md, history in zip(
                policy, result, value, moves_left, outcome, move_delay, histories
            ):
                raw_outputs.append(
                    (
                        p,
                        r,
                        v,
                        ml,
                        o,
                        md,
                        history,
                        self.temperature,
                        return_histories,
                    )
                )
        if raw_outputs:
            del board_tensor
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
        return ChessMovePredictionDataset[AuxiliaryTargetChessMovePrediction](
            predictions=predictions
        )


__all__ = [
    "END_OF_TEXT_PGN_HEADER_PROMPT_TEMPLATE",
    "PipelineInputConfig",
    "TimeInfoPromptStyle",
    "UniMaiaDataset",
    "UniMaiaDatasetWithClockInfo",
    "UniMaiaDatasetWithTimeControl",
    "UniMaiaInputMode",
    "UniMaiaPipeline",
    "UniMaiaPipelineConfig",
    "all_termination_reasons",
]
