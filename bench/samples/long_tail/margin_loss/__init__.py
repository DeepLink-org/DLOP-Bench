# Copyright(c) OpenMMLab. All Rights Reserved.
# Copied from
from bench.common import (
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
        source=SampleSource.SEGMENTBASE2,
        url="https://gitlab.bj.sensetime.com/parrots.fit/segmentbase2/-/blob/master/segmentbase2/models/losses/Margin_loss.py#L45",  # noqa
        tags=[SampleTag.ForLoop, SampleTag.Reduce, SampleTag.ThirdPartyCodes]
    )


def gen_np_args(M):
    def gen_base(M):
        data = np.random.randn(M, 2, 3, 3) * 100
        data = data.astype(np.float32)
        return data

    logit = gen_base(M)
    label = gen_base(M)

    return [logit, label]


register_sample(__name__, get_sample_config, gen_np_args)
