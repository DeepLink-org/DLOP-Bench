from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(4, ), (8, ), (12, )],
        requires_grad=[True, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[
            SampleTag.ForLoop, SampleTag.Reduce, SampleTag.ThirdPartyCodes
        ],
    )


def gen_np_args(N):
    logit = np.random.rand(N, 2, 3, 3).astype(np.float32)
    label = np.random.rand(N, 2, 3, 3).astype(np.float32)
    return [logit, label]


register_sample(__name__, get_sample_config, gen_np_args)
