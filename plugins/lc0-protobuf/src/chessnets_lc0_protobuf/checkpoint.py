"""Read gzipped Lc0 protobuf checkpoints and describe their structure."""

from __future__ import annotations

import gzip
import pathlib

import numpy as np

from . import net_pb2 as pb


def read_checkpoint(path: str | pathlib.Path) -> pb.Net:
    """Parse a gzip-compressed Lc0 ``.pb.gz`` network file."""
    with gzip.open(path, "rb") as stream:
        return pb.Net.FromString(stream.read())


def residual_block_count(checkpoint: pb.Net) -> int:
    """Return the number of residual blocks in a convolutional network."""
    return len(checkpoint.weights.residual)


def filter_count(checkpoint: pb.Net) -> int:
    """Return the convolution width, read from the input block's batch norm.

    Only meaningful for ``LINEAR16`` networks, where each value is stored as
    one 16-bit integer.
    """
    return np.frombuffer(checkpoint.weights.input.bn_means.params, np.uint16).size


__all__ = ["filter_count", "read_checkpoint", "residual_block_count"]
