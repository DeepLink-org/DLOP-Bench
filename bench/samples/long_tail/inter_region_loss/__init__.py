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
        args_cases=[(8, ), (4, ), (6, )],
        requires_grad=[True, False, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SEGMENTBASE2,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.IfElseBranch, \
              SampleTag.ForLoop, SampleTag.Reduce]
    )


def gen_np_args(M):
    def gen_base(M):
        data = np.random.randn(M, 2, 3, 3) * 100
        data = data.astype(np.float32)
        return data

    logit = gen_base(M)
    softlabel = gen_base(M)
    label = gen_base(M)

    return [logit, softlabel, label]

register_sample(__name__, get_sample_config, gen_np_args)
