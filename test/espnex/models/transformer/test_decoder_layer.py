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
from espnex.models.transformer.decoder_layer import DecoderLayer
from espnex.models.utils import make_pad_mask


@mark.parametrize('concat_after', [True, False])
@mark.parametrize('normalize_before', [True, False])
def test_decoder_layer_train_mode(normalize_before, concat_after):
    self_attn = MultiHeadDotProductAttention(num_heads=4, deterministic=False)
    src_attn = MultiHeadDotProductAttention(num_heads=4, deterministic=False)
    ff = PositionwiseFeedForward(hidden_units=512, dropout_rate=.5, deterministic=False)

    decoder = DecoderLayer(self_attn, src_attn, ff, dropout_rate=.5, normalize_before=normalize_before,
                           concat_after=concat_after, deterministic=False)
    keys = random.PRNGKey(0)
    keys = random.split(keys)
    in_shape = 3, 256, 64
    in_lens = np.random.randint(1, 256, size=3, dtype=int)
    enc_shape = 3, 512, 128
    enc_lens = np.random.randint(1, 512, size=3, dtype=int)

    inp = np.random.randn(*in_shape)
    encoded = np.random.randn(*enc_shape)

    decoder_causal_mask = make_causal_mask(inp[..., 0])
    decoder_le_mask = ~make_pad_mask(in_lens, 256)
    decoder_mask = make_attention_mask(decoder_le_mask, decoder_le_mask)
    decoder_mask = combine_masks(decoder_causal_mask, decoder_mask)

    enc_mask = ~make_pad_mask(enc_lens, 512)
    enc_dec_mask = make_attention_mask(decoder_le_mask, enc_mask)

    variables = decoder.init(dict(zip(['dropout', 'params'], keys)), inp, decoder_mask, encoded, enc_dec_mask)
    y = decoder.apply(variables, inp, decoder_mask, encoded, enc_dec_mask, rngs=dict(dropout=keys[0]))
    assert y.shape == (3, 256, 64)


@mark.parametrize('concat_after', [True, False])
@mark.parametrize('normalize_before', [True, False])
def test_decoder_layer_inference_one_step(normalize_before, concat_after):
    """Test `decode` mode (single step inference mode)."""
    self_attn = MultiHeadDotProductAttention(num_heads=4, decode=True, deterministic=False)
    src_attn = MultiHeadDotProductAttention(num_heads=4, deterministic=False)
    ff = PositionwiseFeedForward(hidden_units=512, dropout_rate=.5, deterministic=False)

    decoder = DecoderLayer(self_attn, src_attn, ff, dropout_rate=.5, normalize_before=normalize_before,
                           concat_after=concat_after, deterministic=False)
    keys = random.PRNGKey(0)
    keys = random.split(keys)
    decode_shape = 3, 256, 64
    init_shape = 3, 1, 64
    enc_shape = 3, 512, 128
    enc_lens = np.random.randint(1, 512, size=3, dtype=int)

    inp = np.random.randn(*init_shape)
    encoded = np.random.randn(*enc_shape)

    enc_mask = ~make_pad_mask(enc_lens, 512)  # bs, enc_len
    enc_dec_mask = jnp.expand_dims(enc_mask, [1, 2])  # bs, 1, 1, enc_len

    variables = decoder.init(dict(zip(['dropout', 'params'], keys)), jnp.empty(decode_shape), None, encoded, None)

    cache = variables['cache']
    params = variables['params']

    # test cache shape
    for k, v in cache['self_attn'].items():
        if 'index' in k:
            assert v.shape == ()
        else:
            assert v.shape == (3, 256, 4, 64 // 4)  # expected shape should be (bs, len, n_heads, head_feat_n)

    x = inp
    for _ in range(4):
        x, updated_vars = decoder.apply(
            dict(params=params, cache=cache),
            x, None, encoded, enc_dec_mask,
            mutable='cache',
            rngs=dict(dropout=keys[0])
        )
        cache = updated_vars['cache']
        assert x.shape == init_shape
