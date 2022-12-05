import jax

from espnex.layers.log_mel import LogMel
from espnet2.layers.log_mel import LogMel as TLogMel
from pytest import mark
from test.espnex.utils import *


@mark.parametrize('fs', [8000, 16000, 32000])
@mark.parametrize('n_fft', [64, 256, 512])
@mark.parametrize('n_mels', [40, 80])
@mark.parametrize('htk', [True, False])
def test_log_mel(
        fs,
        n_fft,
        n_mels,
        htk
):
    key, = split(init_key, 1)
    x = uniform(key, [5, 257, n_fft // 2 + 1], float, 0, 1000)
    key, = split(key, 1)
    ilens = randint(key, [5], 1, 257)
    log_mel = LogMel(fs=fs, n_mels=n_mels, htk=htk)

    call = jax.jit(lambda x, ilens: log_mel(x, ilens))
    y, olens = call(x, ilens)
    y, olens = map(jax.device_get, (y, olens))

    x, ilens = map(j2t, [x, ilens])
    tlog_mel = TLogMel(fs=fs, n_fft=n_fft, n_mels=n_mels, htk=htk)
    ty, tolens = tlog_mel(x, ilens)
    ty, tolens = ty.numpy(), tolens.numpy()

    assert_allclose(y, ty, rtol=1e-4, atol=1e-4)
    assert_equal(olens, tolens)







