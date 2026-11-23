# SPDX-License-Identifier: MIT
# Portions adapted from Allie (https://github.com/ippolito-cmu/allie),
# Copyright (c) 2025 Yiming Zhang. See LICENSES/Allie-MIT.txt.
# Modified for ChessNets by Sherman Siu; modifications are under the same license.

"""Small inference helpers for Allie models."""

from __future__ import annotations

import torch

Batch = dict[str, torch.Tensor]


def to_device(batch: Batch, device: str, blocking: bool = False) -> Batch:
    """Move a tensor batch to a device, using pinned memory for CUDA."""
    if not isinstance(batch, dict):
        raise TypeError(f"Expected a tensor mapping, got {type(batch).__name__}")
    if device.startswith("cuda"):
        return {
            key: value.pin_memory().to(device, non_blocking=not blocking)
            for key, value in batch.items()
        }
    return batch


def count_params(model: torch.nn.Module) -> int:
    """Return the total number of model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = ["Batch", "count_params", "to_device"]
