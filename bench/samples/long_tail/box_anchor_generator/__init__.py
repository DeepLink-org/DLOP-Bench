from long_tail_bench.common import SampleConfig, register_sample
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(32, 32), (32, 24), (24, 24)],
        requires_grad=[False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
    )


def gen_np_args(H, W):
    a = 3
    stride = 2
    shape = np.array([[H], [W], [H * W * a], [stride]])
    return [shape]


register_sample(__name__, get_sample_config, gen_np_args)
