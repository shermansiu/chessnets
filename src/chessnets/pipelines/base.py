"""Shared pipeline contracts."""

from __future__ import annotations

import abc
import collections.abc
import dataclasses
import typing as tp

# These names are deliberately strings.  They are part of the public error
# messages and are also useful when inspecting a pipeline from a notebook.
PipelineInputName = str


_REQUIRED_INPUT_NAMES = {
    "requires_pgns": "pgns",
    "requires_fens": "fens",
    "requires_prompts": "prompts",
    "requires_time_controls": "time_controls",
    "requires_elos": "elos",
    "requires_elos_self_oriented": "elo_self_oriented",
    "requires_clock_times": "clock_times",
    "requires_move_seconds": "move_seconds",
}


_INPUT_ALIASES = {
    "pgn": "pgns",
    "fens": "fens",
    "prompt": "prompts",
    "time_control": "time_controls",
    "elo": "elos",
    "white_elo": "white_elos",
    "black_elo": "black_elos",
    "elo_self": "elo_self_oriented",
    "elo_oppo": "elo_opponent",
    "clock_time": "clock_times",
    "white_clock_time": "white_clock_times",
    "black_clock_time": "black_clock_times",
    "move_second": "move_seconds",
}

_KNOWN_INPUT_NAMES = frozenset(_REQUIRED_INPUT_NAMES.values()) | frozenset(
    {
        # Retain the removed singular spelling as a domain-looking input so
        # pipeline ``**kwargs`` paths reject it instead of silently ignoring it.
        "fen",
        "white_elos",
        "black_elos",
        "elo_opponent",
        "white_clock_times",
        "black_clock_times",
    }
)


def _canonical_input_name(name: str) -> str:
    return _INPUT_ALIASES.get(name, name)


def _canonicalize_provided(provided: dict[str, tp.Any]) -> dict[str, tp.Any]:
    canonical: dict[str, tp.Any] = {}
    original_names: dict[str, str] = {}
    for name, value in provided.items():
        canonical_name = _canonical_input_name(name)
        if canonical_name in canonical and original_names[canonical_name] != name:
            raise ValueError(
                "Pipeline input was provided more than once using aliases: "
                f"{original_names[canonical_name]!r} and {name!r}"
            )
        canonical[canonical_name] = value
        original_names[canonical_name] = name
    return canonical


def extract_domain_inputs(provided: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Extract domain-input-looking values from a ``**kwargs`` mapping.

    Text backends legitimately accept additional generation kwargs.  Pipeline
    implementations can use this helper to ensure a domain input accidentally
    supplied through ``**kwargs`` is still checked against the contract.
    """
    return {
        name: value
        for name, value in provided.items()
        if _canonical_input_name(name) in _KNOWN_INPUT_NAMES
    }


def _is_sequence(value: tp.Any) -> bool:
    return isinstance(value, collections.abc.Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


@dataclasses.dataclass(frozen=True)
class PipelineInputConfig:
    requires_pgns: bool
    requires_fens: bool
    requires_prompts: bool
    requires_time_controls: bool = False
    requires_elos: bool = False
    requires_elos_self_oriented: bool = False
    requires_clock_times: bool = False
    requires_move_seconds: bool = False
    # ``None`` preserves the permissive behavior of the original contract for
    # third-party pipelines.  Built-in pipelines provide this explicitly so
    # unsupported domain inputs can be rejected.
    accepted_inputs: frozenset[str] | None = None
    required_inputs_extra: frozenset[str] = frozenset()
    paired_inputs: tuple[frozenset[str], ...] = ()

    def __post_init__(self) -> None:
        if self.accepted_inputs is not None:
            accepted = frozenset(
                _canonical_input_name(name) for name in self.accepted_inputs
            )
            object.__setattr__(self, "accepted_inputs", accepted)
        object.__setattr__(
            self,
            "required_inputs_extra",
            frozenset(
                _canonical_input_name(name) for name in self.required_inputs_extra
            ),
        )
        pairs = tuple(
            frozenset(_canonical_input_name(name) for name in pair)
            for pair in self.paired_inputs
        )
        object.__setattr__(self, "paired_inputs", pairs)
        if self.accepted_inputs is not None:
            missing = self.required_inputs - self.accepted_inputs
            if missing:
                raise ValueError(
                    "Required pipeline input(s) must also be accepted: "
                    + ", ".join(sorted(missing))
                )
            unsupported_pair = sorted(
                name
                for pair in self.paired_inputs
                for name in pair
                if name not in self.accepted_inputs
            )
            if unsupported_pair:
                raise ValueError(
                    "Paired pipeline input(s) must be accepted: "
                    + ", ".join(unsupported_pair)
                )

    @property
    def required_inputs(self) -> frozenset[str]:
        required = (
            frozenset(
                name
                for attribute, name in _REQUIRED_INPUT_NAMES.items()
                if getattr(self, attribute)
            )
            | self.required_inputs_extra
        )
        # Keep the old aggregate flags meaningful for callers that inspect
        # them, while component-wise contracts remain the authoritative names.
        if {"white_elos", "black_elos"}.issubset(required):
            required = required - {"elos"}
        if {"white_clock_times", "black_clock_times"}.issubset(required):
            required = required - {"clock_times"}
        return required

    @property
    def allowed_inputs(self) -> frozenset[str] | None:
        """Return accepted names, or ``None`` for legacy permissive configs."""
        if self.accepted_inputs is None:
            return None
        return self.accepted_inputs

    @classmethod
    def from_inputs(
        cls,
        required: tp.Iterable[str],
        *,
        optional: tp.Iterable[str] = (),
        paired: tp.Iterable[tp.Iterable[str]] = (),
    ) -> PipelineInputConfig:
        """Construct a contract from canonical public input names.

        The legacy boolean fields remain populated so old callers inspecting
        ``requires_*`` continue to work.  New code should prefer this factory.
        """
        required_names = frozenset(_canonical_input_name(name) for name in required)
        optional_names = frozenset(_canonical_input_name(name) for name in optional)
        accepted = required_names | optional_names
        kwargs = {
            attribute: name in required_names
            for attribute, name in _REQUIRED_INPUT_NAMES.items()
        }
        if {"white_elos", "black_elos"}.issubset(required_names):
            kwargs["requires_elos"] = True
        if {"white_clock_times", "black_clock_times"}.issubset(required_names):
            kwargs["requires_clock_times"] = True
        return cls(
            **kwargs,
            accepted_inputs=accepted,
            required_inputs_extra=frozenset(
                required_names - frozenset(_REQUIRED_INPUT_NAMES.values())
            ),
            paired_inputs=tuple(
                frozenset(_canonical_input_name(name) for name in pair)
                for pair in paired
            ),
        )

    def validate(self, *, strict: bool = True, **provided: tp.Any) -> None:
        """Validate required, accepted, and paired inputs.

        ``strict`` only affects contracts created with ``accepted_inputs``.  A
        legacy config remains permissive, which avoids breaking external
        pipelines that subclass :class:`PipelineInputConfig`.
        """
        canonical_provided = _canonicalize_provided(provided)
        field_requirements = self.required_inputs

        if strict and self.allowed_inputs is not None:
            unsupported = sorted(
                name
                for name, value in canonical_provided.items()
                if value is not None and name not in self.allowed_inputs
            )
            if unsupported:
                accepted = ", ".join(sorted(self.allowed_inputs)) or "none"
                raise ValueError(
                    "Unsupported pipeline input(s): "
                    + ", ".join(unsupported)
                    + f". Accepted input(s): {accepted}"
                )

        missing = [
            name
            for name in sorted(field_requirements)
            if canonical_provided.get(name) is None
        ]
        if missing:
            raise ValueError(
                "Missing required pipeline input(s): " + ", ".join(missing)
            )

        for pair in self.paired_inputs:
            present = [canonical_provided.get(name) is not None for name in pair]
            if any(present) and not all(present):
                missing_pair = sorted(
                    name for name, is_present in zip(pair, present) if not is_present
                )
                raise ValueError(
                    "Pipeline inputs must be provided together: "
                    + ", ".join(sorted(pair))
                    + ". Missing: "
                    + ", ".join(missing_pair)
                )

    def normalize(
        self,
        *,
        primary: str,
        broadcast: tp.Iterable[str] = (),
        **provided: tp.Any,
    ) -> dict[str, tp.Any]:
        """Validate and normalize a batch of scalar/sequence inputs.

        Scalars are converted to one-item lists.  Auxiliary values are not
        silently broadcast: callers must opt in by naming them in
        ``broadcast``.  This catches the common mistake of passing one prompt
        for a multi-PGN batch.
        """
        primary = _canonical_input_name(primary)
        canonical_provided = _canonicalize_provided(provided)
        self.validate(**canonical_provided)
        if canonical_provided.get(primary) is None:
            raise ValueError(f"Primary pipeline input {primary!r} is required")

        normalized: dict[str, tp.Any] = {}
        for name, value in canonical_provided.items():
            if value is None:
                normalized[name] = None
            elif _is_sequence(value):
                normalized[name] = list(value)
            else:
                normalized[name] = [value]

        batch_size = len(normalized[primary])
        if batch_size == 0:
            raise ValueError(f"Pipeline input {primary!r} must not be empty")
        broadcast_names = {_canonical_input_name(name) for name in broadcast}
        for name, value in normalized.items():
            if value is None or name == primary:
                continue
            if len(value) == batch_size:
                continue
            if name in broadcast_names and len(value) == 1:
                normalized[name] = value * batch_size
                continue
            raise ValueError(
                f"Pipeline input {name!r} has {len(value)} item(s), expected "
                f"{batch_size} to match {primary!r}"
            )
        return normalized


def bind_positional_inputs(
    order: tp.Sequence[str],
    positional: tp.Sequence[tp.Any],
    keywords: tp.Mapping[str, tp.Any],
) -> dict[str, tp.Any]:
    """Assign inputs passed positionally after the PGNs to the names in ``order``.

    Pipelines whose inputs depend on a configured mode use this so that callers
    may pass inputs in their canonical order (``pgns, prompts, time_controls,
    white_elos, black_elos``), as ChessEval does, or by keyword.
    """
    if len(positional) > len(order):
        expected = ", ".join(order) or "none"
        raise TypeError(
            f"Expected at most {len(order)} positional input(s) after the PGNs "
            f"({expected}), got {len(positional)}"
        )
    bound = dict(keywords)
    for name, value in zip(order, positional):
        if bound.get(name) is not None:
            raise TypeError(f"Pipeline input {name!r} was given twice")
        bound[name] = value
    return bound


BatchSize = int | tp.Literal["auto"]
"""A batch size, or ``"auto"`` to find one that fits in memory."""


class Pipeline(abc.ABC):
    auto_batch_sizes: tp.ClassVar[tp.Mapping[str, int]] = {"cpu": 16, "cuda": 16}
    """The batch size ``batch_size="auto"`` starts from, by device type.

    Device types without an entry use the ``cpu`` value. Pipelines override
    this with sizes suited to their models.
    """

    last_batch_size: int | None = None
    """The batch size the most recent call used."""

    def __init__(self, input_config: PipelineInputConfig | None = None):
        self.input_config = input_config or self.default_input_config()

    @staticmethod
    def default_input_config() -> PipelineInputConfig:
        return PipelineInputConfig.from_inputs(required=("pgns",))

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: tp.Any, **kwargs: tp.Any
    ) -> tp.Self:
        """Construct a pipeline from a pretrained model or hosted model ID.

        Concrete pipelines that can load a model must override this method.
        Pipelines whose constructor receives an already-created backend model
        intentionally keep that dependency-injection API and should provide a
        backend-specific implementation when one exists.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not define a pretrained loader; construct it "
            "with an already-loaded model instead."
        )

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def make_prediction(self, pgn: str | list[str], *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        """Predict, as :meth:`make_prediction` does.

        ``batch_size="auto"`` starts from :attr:`auto_batch_sizes` for the
        pipeline's device and halves the batch size whenever a batch runs out
        of memory, retrying the call. The size that worked is kept for later
        calls, and :attr:`last_batch_size` records it.
        """
        batch_size = kwargs.get("batch_size")
        if batch_size != "auto":
            if isinstance(batch_size, int):
                self.last_batch_size = batch_size
            return self.make_prediction(*args, **kwargs)
        return self._make_prediction_with_auto_batch_size(*args, **kwargs)

    def _make_prediction_with_auto_batch_size(self, *args, **kwargs):
        import inspect
        import logging

        from .pipeline_utils import free_memory, is_out_of_memory_error

        if "batch_size" not in inspect.signature(self.make_prediction).parameters:
            kwargs.pop("batch_size")
            return self.make_prediction(*args, **kwargs)

        batch_size = getattr(self, "_found_batch_size", None)
        if batch_size is None:
            sizes = self.auto_batch_sizes
            batch_size = sizes.get(self._device_type(), sizes.get("cpu", 16))
        while True:
            try:
                result = self.make_prediction(
                    *args, **{**kwargs, "batch_size": batch_size}
                )
            except Exception as error:
                if batch_size == 1 or not is_out_of_memory_error(error):
                    raise
            else:
                self._found_batch_size = batch_size
                self.last_batch_size = batch_size
                return result
            # Free the failed batch's memory, now that the exception holding
            # its tensors is gone, and try again with half as many items.
            free_memory(collect_garbage=True)
            logging.getLogger(__name__).warning(
                "%s ran out of memory with a batch size of %d; retrying with %d",
                type(self).__name__,
                batch_size,
                batch_size // 2,
            )
            batch_size //= 2

    def _device_type(self) -> str:
        """``cuda``, ``mps`` or ``cpu``: where the pipeline's model runs."""
        device = getattr(self, "device", None)
        if device is None:
            model = getattr(self, "model", None)
            device = getattr(model, "device", None)
            if device is None and callable(getattr(model, "parameters", None)):
                try:
                    device = next(iter(model.parameters())).device
                except (StopIteration, TypeError):
                    device = None
        return str(device or "cpu").split(":")[0]

    def notebook_display_prediction(self, pgn: str | list[str], *args, **kwargs):
        prediction = self.make_prediction(pgn, *args, **kwargs)
        prediction.notebook_display()


__all__ = [
    "BatchSize",
    "Pipeline",
    "PipelineInputName",
    "PipelineInputConfig",
    "bind_positional_inputs",
    "extract_domain_inputs",
]
