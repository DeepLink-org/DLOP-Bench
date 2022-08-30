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
        args_cases=[(300, ), (600, ), (800, )],
        requires_grad=[True, False, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMPOSE,
        url="https://github.com/open-mmlab/mmpose/blob/c85159da60f25bc6bbab763c37c2be1e49e4052f/mmpose/models/losses/mse_loss.py#L9",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.ForLoop, \
              SampleTag.IfElseBranch, SampleTag.Reduce]
    )


def gen_np_args(N):
    def gen_base(num):
        data = np.random.randn(num, 4) * 100
        data = data.astype(np.float32)
        return data

    bboxes = gen_base(N)
    gt = gen_base(N)
    target_weight = gen_base(N)
    return [bboxes, gt, target_weight]


register_sample(__name__, get_sample_config, gen_np_args)
