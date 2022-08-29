from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(80, 13), (160, 13), (160, 26)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[SampleTag.Broadcast, SampleTag.ViewAttribute],
    )


def gen_np_args(h, w):
    c = 3
    img = np.random.randn(h, w, c).astype(np.float32)
    mean = np.empty([3], dtype=np.float32, order='C')
    mean[0] = 1
    mean[1] = 1
    mean[2] = 1
    scale = np.empty([3], dtype=np.float32, order='C')
    scale[0] = 5
    scale[0] = 5
    scale[0] = 5
    return [img, mean, scale]


register_sample(__name__, get_sample_config, gen_np_args)
