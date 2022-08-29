from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, 4), (4800, 4), (5000, 4)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[SampleTag.IfElseBranch, SampleTag.Reduce]
    )


def gen_np_args(N, K):
    boxes = np.random.randn(N, K)
    idxs = np.random.randint(100, (N, ))
    return [boxes, idxs]


register_sample(__name__, get_sample_config, gen_np_args)
