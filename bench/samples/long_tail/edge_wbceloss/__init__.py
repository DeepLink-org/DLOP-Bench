# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(2, ), (4, ), (8, )],
        requires_grad=[True, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SEGMENTBASE2,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.Reduce]
    )


def gen_np_args(N):
    logit = np.random.randn(N, 1, 3, 3)
    logit = logit.astype(np.float32)
    label = np.random.randn(N, 2, 3, 3)
    label = label.astype(np.float32)
    return [logit, label]


register_sample(__name__, get_sample_config, gen_np_args)
