import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import (Dense, Dropout, LayerNorm, Module, cond, MultiHeadDotProductAttention, compact,
                        make_causal_mask, combine_masks, make_attention_mask)
from flax.linen.module import compact, merge_param
from jax import Array
from jax import random
import numpy as np
from pytest import mark
from jax.tree_util import tree_flatten

from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnex.models.transformer.multi_layer_conv import MultiLayerConv1d, Conv1dLinear
from espnex.models.utils import make_pad_mask
from espnex.models.transformer.decoder import Decoder


@mark.parametrize('concat_after', [True, False])
@mark.parametrize('normalize_before', [True, False])
def test_decoder_train_mode(normalize_before, concat_after):
    voc_size = 100

    decoder = Decoder(voc_size, linear_units=512, num_blocks=3, normalize_before=normalize_before,
                      concat_after=concat_after, deterministic=False)
    in_shape = 3, 256
    in_lens = np.random.randint(1, 256, size=3, dtype=int)
    keys = random.PRNGKey(0)
    keys = random.split(keys)
    enc_shape = 3, 512, 128
    enc_lens = np.random.randint(1, 512, size=3, dtype=int)

    inp = np.random.randint(0, voc_size, in_shape)
    encoded = np.random.randn(*enc_shape)

    decoder_causal_mask = make_causal_mask(inp)
    decoder_le_mask = ~make_pad_mask(in_lens, 256)
    decoder_mask = make_attention_mask(decoder_le_mask, decoder_le_mask)
    decoder_mask = combine_masks(decoder_causal_mask, decoder_mask)

    enc_mask = ~make_pad_mask(enc_lens, 512)
    enc_dec_mask = make_attention_mask(decoder_le_mask, enc_mask)

    variables = decoder.init(dict(zip(['dropout', 'params'], keys)), inp, encoded, decoder_mask, enc_dec_mask)
    y = decoder.apply(variables, inp, encoded, decoder_mask, enc_dec_mask, rngs=dict(dropout=keys[0]))
    assert y.shape == (3, 256, voc_size)


@mark.parametrize('concat_after', [True, False])
@mark.parametrize('normalize_before', [True, False])
def test_decoder_inference_one_step(normalize_before, concat_after):
    voc_size = 100

    decoder = Decoder(voc_size, linear_units=512, num_blocks=3, normalize_before=normalize_before,
                      concat_after=concat_after, deterministic=False)
    decode_shape = 3, 256
    init_shape = 3, 1

    keys = random.PRNGKey(0)
    keys = random.split(keys)
    enc_shape = 3, 512, 128
    enc_lens = np.random.randint(1, 512, size=3, dtype=int)

    inp = np.zeros(init_shape).astype(int)  # init_shape
    encoded = np.random.randn(*enc_shape)

    enc_mask = ~make_pad_mask(enc_lens, 512)  # bs, enc_len
    cross_attn_mask = jnp.expand_dims(enc_mask, [1, 2])  # bs, 1, 1, enc_len

    variables = decoder.init(dict(zip(['dropout', 'params'], keys)), jnp.empty(decode_shape, dtype=int), encoded, decode=True)

    cache = variables['cache']
    params = variables['params']

    x = inp
    for _ in range(4):
        x, updated_vars = decoder.apply(
            dict(params=params, cache=cache),
            x, encoded, None, cross_attn_mask,
            mutable='cache',
            rngs=dict(dropout=keys[0])
        )
        cache = updated_vars['cache']
        assert x.shape == init_shape + (voc_size,)
        x = jnp.argmax(x, axis=-1)

