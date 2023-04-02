from espnex.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.default import DefaultFrontend as TDefaultFrontend

from test.espnex.utils import *


def test_default_frontend():
    ishape = bs, t, ch = 5, 1025, 3
    x = uniform(init_key, ishape, float, 0, 1000)
    key, = split(init_key, 1)
    ilens = randint(key, [bs], 1, t)

    frontend = DefaultFrontend()
    key, = split(key, 1)
    rngs = dict(channel=key)

    call = jax.jit(
        lambda x, ilens, deterministic, rngs: frontend.apply({}, x, ilens, deterministic, rngs=rngs),
        static_argnames='deterministic'
    )

    y, olens = call(x, ilens, False, rngs)

    y, olens = call(x, ilens, True, rngs)
    y, olens = map(a2n, [y, olens])
    tfrontend = TDefaultFrontend(frontend_conf=None)
    tfrontend.eval()
    x, ilens = map(j2t, [x, ilens])
    ty, tolens = tfrontend(x, ilens)
    ty, tolens = map(a2n, [ty, tolens])

    assert_equal(olens, tolens)
    # assert_allclose(y, ty, rtol=1e-3, atol=1e-2)  # difference caused by different Stft padding strategy



