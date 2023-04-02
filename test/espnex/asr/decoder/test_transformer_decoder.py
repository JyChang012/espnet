from functools import partial

import jax

from espnex.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder as TorchModel
from test.espnex.utils import *


def test_transformer_decoder():
    mdl = TransformerDecoder(130, 12, layer_drop_rate=0.1)
    x = np.random.randint(0, 130, [2, 233], dtype=int)
    x_len = np.random.randint(0, 233, [2], dtype=int)
    e = np.random.random([2, 500, 12])
    e_len = np.random.randint(0, 500, [2], dtype=int)
    var = jax.jit(partial(mdl.init,
                          deterministic=True,
                          decode=False))(dict(params=init_key),
                                         x, x_len,
                                         e,
                                         e_len)
    params = var['params']

    # compare # of params
    tmdl = TorchModel(130, 12)
    assert compare_params(tmdl, params), 'Diff # of parameters!'

    apply_fn = jax.jit(partial(mdl.apply, deterministic=False, decode=False))
    y, y_len = apply_fn(dict(params=params),
                        x,
                        x_len,
                        e,
                        e_len,
                        rngs=dict(dropout=init_key,
                                  skip_layer=init_key))
    assert_equal(y_len, x_len)
    assert_equal(y.shape[:-1], x.shape)
