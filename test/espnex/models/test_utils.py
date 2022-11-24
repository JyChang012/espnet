import jax
import jax.numpy as jnp
from numpy.testing import assert_equal
from pytest import mark

from espnex.models.utils import make_pad_mask


@mark.parametrize(
    "lengths, maxlen, mask",
    [
        [
            jnp.array([2, 3, 1, 5]),
            6,
            jnp.array(
                [
                    [False, False, True, True, True, True],
                    [False, False, False, True, True, True],
                    [False, True, True, True, True, True],
                    [False, False, False, False, False, True],
                ]
            ),
        ]
    ],
)
def test_make_pad_mask(lengths, maxlen, mask):
    ret = make_pad_mask(lengths, maxlen)
    assert_equal(jax.device_get(ret), jax.device_get(mask))
