# __init__.py

from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(20, 5), (40, 8), (60, 12)],
        requires_grad=[False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN
    )


def gen_np_args(M, N):
    boxes = np.random.randn(M, N)
    boxes = boxes.astype(np.float32)
    return [boxes]


register_sample(__name__, get_sample_config, gen_np_args)
