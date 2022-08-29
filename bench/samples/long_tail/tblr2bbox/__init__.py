from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, ), (4000, ), (5000, )],
        requires_grad=[False] * 2,
        backward=[False],
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[
            SampleTag.IfElseBranch, SampleTag.ViewAttribute,
            SampleTag.Broadcast
        ],
    )


def gen_np_args(N):
    priors = np.random.randn(N, 4).astype(np.float32)
    tblr = np.random.randn(N, 4).astype(np.float32)
    return [priors, tblr]


register_sample(__name__, get_sample_config, gen_np_args)
