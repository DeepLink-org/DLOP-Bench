# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np

def get_sample_config():
    return SampleConfig(
        args_cases=[(30, ), (40, ), (60, )],
        requires_grad=[False]*2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[
            SampleTag.ForLoop, SampleTag.ThirdPartyCodes,
            SampleTag.AdvancedIndexing, SampleTag.Reduce
        ],
    )


def gen_np_args(N):
    def gen_base(num):
        data = np.random.randn(num, num, 1)
        data = data.astype(np.float)
        return data

    logit = gen_base(N)
    target = np.random.randn(N, 1)
    target = target.astype(np.float)
    return [logit, target]


register_sample(__name__, get_sample_config, gen_np_args)
