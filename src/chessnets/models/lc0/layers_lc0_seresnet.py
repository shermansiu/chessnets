"""Building blocks for convolutional (SE-ResNet) Lc0 networks.

Original ChessNets implementation, written from the published Lc0 network
structure (``NETWORK_SE_WITH_HEADFORMAT``) and validated numerically against
converted Maia-1 checkpoints. Submodule attribute names are fixed by the
ChessNets safetensors checkpoint layout, so they must not be renamed without a
checkpoint migration.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from . import policy_map

BOARD_SIZE = 64
POLICY_PLANES = 80


class ConvBatchNormReLU(nn.Module):
    """Bias-free square convolution, batch normalization, then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.batchnorm(self.conv_layer(x)))


class SqueezeExcite(nn.Module):
    """Lc0-style squeeze-and-excitation with a learned scale *and* shift.

    The globally pooled features pass through a bottleneck MLP that emits
    ``2 * channels`` values: the first half is a sigmoid gate on the input and
    the second half is added to it as a per-channel bias.
    """

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden = channels // reduction
        self.squeeze = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU())
        self.expand = nn.Linear(hidden, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = x.mean(dim=(-2, -1))
        gate, shift = self.expand(self.squeeze(summary)).chunk(2, dim=1)
        return x * gate.sigmoid()[..., None, None] + shift[..., None, None]


class SEResidualBlock(nn.Module):
    """Two 3x3 conv/batch-norm layers with an SE unit on the residual branch."""

    def __init__(self, channels: int, se_reduction: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(channels)
        self.squeeze_excite = SqueezeExcite(channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = F.relu(self.batch_norm1(self.conv1(x)))
        branch = self.squeeze_excite(self.batch_norm2(self.conv2(branch)))
        return F.relu(x + branch)


class ConvPolicyHead(nn.Module):
    """Predict 80 move planes per square and gather the 1858 Lc0 policy logits."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = ConvBatchNormReLU(channels, channels, 3)
        self.conv = nn.Conv2d(channels, POLICY_PLANES, 3, padding=1)
        self.register_buffer(
            "policy_indices",
            torch.from_numpy(policy_map.convolution_policy_indices()),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        move_planes = self.conv(self.conv_block(x))
        return move_planes.flatten(1)[:, self.policy_indices]


class ConvValueHead(nn.Module):
    """1x1 channel projection followed by a two-layer MLP (the WDL value head).

    ``fc2`` is the hidden layer and ``fc_out`` the output layer; both names are
    fixed by the checkpoint layout.
    """

    def __init__(
        self,
        in_channels: int,
        projection_channels: int,
        hidden_size: int,
        out_features: int,
    ):
        super().__init__()
        self.conv_block = ConvBatchNormReLU(in_channels, projection_channels, 1)
        self.fc2 = nn.Linear(projection_channels * BOARD_SIZE, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_block(x).flatten(1)
        return self.fc_out(F.relu(self.fc2(features)))


__all__ = [
    "ConvBatchNormReLU",
    "ConvPolicyHead",
    "ConvValueHead",
    "SEResidualBlock",
    "SqueezeExcite",
]
