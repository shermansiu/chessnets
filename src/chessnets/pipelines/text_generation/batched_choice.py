"""Experimental batched generation with a different choice for each row.

Outlines applies one logits processor to a whole batch, and its Outlines Core
processor builds every row's guide from the same index. This processor keeps
one Outlines Core guide per row, each from its own index, so one batched
``generate`` call can restrict every position to its own legal moves.

It is written against Outlines' public ``OutlinesLogitsProcessor`` base class
and Outlines Core's ``Guide``, ``Index`` and bitmask kernels, not Outlines'
private processor internals. Only PyTorch tensors are supported.
"""

from __future__ import annotations

import typing as tp

import torch
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
from outlines_core import Guide, Index
from outlines_core.kernels.torch import (
    apply_token_bitmask_inplace,
    fill_next_token_bitmask,
)


class PerRowChoiceLogitsProcessor(OutlinesLogitsProcessor):
    """Constrain row ``i`` of a batch with ``indices[i]``.

    The guides are advanced exactly as Outlines' ``OutlinesCoreLogitsProcessor``
    advances its own, so a batch of one behaves like it.
    """

    def __init__(self, indices: tp.Sequence[Index]):
        self.indices = list(indices)
        super().__init__("torch")
        self.reset()

    def reset(self) -> None:
        self._guides: list[Guide] | None = None
        self._bitmask: torch.Tensor | None = None

    def process_logits(self, input_ids: torch.Tensor, logits: torch.Tensor):
        batch_size, vocab_size = input_ids.shape[0], logits.shape[1]
        if batch_size != len(self.indices):
            raise ValueError(
                f"Expected a batch of {len(self.indices)} rows, got {batch_size}"
            )
        if self._guides is None:
            self._guides = [Guide(index) for index in self.indices]
            self._bitmask = torch.full(
                (batch_size, (vocab_size + 31) // 32),
                -1,
                dtype=torch.int32,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            last_tokens = input_ids[:, -1].tolist()
            for guide, token_id in zip(self._guides, last_tokens):
                # Same condition as OutlinesCoreLogitsProcessor (outlines_core
                # issue #227): a finished guide only advances on a token it
                # accepts, such as padding after EOS.
                if not guide.is_finished() or guide.accepts_tokens([token_id]):
                    guide.advance(token_id=token_id, return_tokens=False)

        assert self._bitmask is not None
        for row, guide in enumerate(self._guides):
            fill_next_token_bitmask(guide, self._bitmask[row : row + 1])
        apply_token_bitmask_inplace(
            logits, self._bitmask.to(logits.device, non_blocking=True)
        )
        return logits


__all__ = ["PerRowChoiceLogitsProcessor"]
