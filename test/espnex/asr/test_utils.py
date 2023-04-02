from espnex.asr.utils import ASRErrorCalculator
import numpy as np
from pytest import mark


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
            [
                (1,),
                (2,),
                (3, 2)
            ],
            [1, 1, 2]
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
            [
                (1, 2, 2, 3, 4, 5, 2),
                (1, 2, 2, 3),
                (1, 2, 2, 2, 2, 2, 2),
                (),
                (3,)
            ],
            [7, 4, 7, 0, 1]

        ]
    ]
)
def test_error_calculator(
        input: np.ndarray,
        input_lengths: np.ndarray,
        ref_decoded,
        ref_decoded_lengths
):
    ec = ASRErrorCalculator(['a', 'b', 'c'], blank_id=0, space_id=1)
    decoded = ec.batch_ctc_decode(input, input_lengths)
    assert decoded == ref_decoded
