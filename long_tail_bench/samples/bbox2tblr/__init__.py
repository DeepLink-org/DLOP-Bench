from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, 4), (4000, 4), (5000, 4)],
        requires_grad=[False] * 2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[SampleTag.IfElseBranch, SampleTag.ViewAttribute],
    )


def gen_np_args(M, N):
    priors = np.random.randn(M, N)
    priors = priors.astype(np.float32)
    gts = np.random.randn(M, N)
    gts = gts.astype(np.float32)
    return [priors, gts]


register_sample(__name__, get_sample_config, gen_np_args)
