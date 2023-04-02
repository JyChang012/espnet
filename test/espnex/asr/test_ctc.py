import jax
import numpy as np
from jax import device_get
from pytest import mark
from numpy.testing import assert_equal

from espnex.asr.ctc import ctc_decode


@mark.parametrize(
    'input, input_lengths, ref_decoded, ref_decoded_lengths',
    [
        # test case 0
        [
            np.array([
                [1, 0],
                [2, 2],
                [3, 2]
            ]),
            np.array([1, 2, 2]),
            np.array([
                [1, -1],
                [2, -1],
                [3, 2]
            ]),
            np.array([1, 1, 2])
        ],

        # task case 1
        [
            np.array([
                [1, 1, 2, 2, 2, 0, 2, 0, 0, 0, 3, 4, 4, 4, 5, 0, 2],
                [1, 1, 2, 2, 2, 0, 0, 2, 0, 0, 3, 4, 4, 4, 5, 0, 2],
                [1, 0, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2],
                [0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2],
                [3, 0, 2, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2],
            ]),
            np.array(
                [17, 11, 17, 1, 1]
            ),
            np.array([
                [1, 2, 2, 3, 4, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1] * 17,
                [3] + [-1] * 16
            ]),
            np.array([7, 4, 7, 0, 1])

        ]
    ]
)
def test_ctc_decode_function(
        input,
        input_lengths,
        ref_decoded,
        ref_decoded_lengths
):
    ctc_decode_jit = jax.jit(ctc_decode)
    decoded, decoded_lengths = ctc_decode_jit(input, input_lengths)
    decoded, decoded_lengths = device_get([decoded, decoded_lengths])
    assert_equal(decoded, ref_decoded)
    assert_equal(decoded_lengths, ref_decoded_lengths)

