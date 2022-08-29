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
        args_cases=[(128, ), (256, ), (512, )],
        requires_grad=[True, False, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SEGMENTBASE2,
        url="https://gitlab.bj.sensetime.com/parrots.fit/segmentbase2/-/blob/master/segmentbase2/models/losses/ifv_loss.py#L18",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.Reduce,\
              SampleTag.ForLoop, SampleTag.IfElseBranch]
    )


def gen_np_args(N):
    def gen_base(num):
        data = np.random.randn(num, 2, 3, 3) * 100
        data = data.astype(np.float32)
        return data

    logit = gen_base(N)
    softlabel = gen_base(N)
    label = gen_base(N)
    return [logit, softlabel, label]


register_sample(__name__, get_sample_config, gen_np_args)
