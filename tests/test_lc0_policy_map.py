import numpy as np

from chessnets.models.lc0_policy_map import (
    arc_position_encoding,
    attention_policy_indices,
    convolution_policy_indices,
)
from chessnets.utils.decode_policy import MOVES


def test_generated_policy_move_vocabulary_is_stable():
    assert len(MOVES) == len(set(MOVES)) == 1858
    assert MOVES[:3] == ("a1b1", "a1c1", "a1d1")
    assert MOVES[-3:] == ("h7h8q", "h7h8r", "h7h8b")


def test_policy_maps_have_expected_ranges_and_shapes():
    convolution = convolution_policy_indices()
    attention = attention_policy_indices()
    position = arc_position_encoding()

    assert convolution.shape == attention.shape == (1858,)
    assert convolution.dtype == attention.dtype == np.int64
    assert convolution.min() >= 0
    assert convolution.max() < 80 * 8 * 8
    assert attention.min() >= 0
    assert attention.max() < 64 * 64 + 8 * 24
    assert position.shape == (1, 64, 64)
    assert np.all(np.diag(position[0]) == -1)
