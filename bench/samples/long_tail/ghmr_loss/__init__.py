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
        args_cases=[(3000, ), (2000, ), (4000, )],
        requires_grad=[True, False, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/2294badd86b0bc83e49692187a4639b69f2ec4b8/mmdet/models/losses/ghm_loss.py#L122",  # noqa
        tags=[SampleTag.ViewAttribute, \
              SampleTag.IfElseBranch, SampleTag.ForLoop]
    )


def gen_np_args(N):
    def gen_base(num):
        data = np.random.randn(num, 4) * 100
        data = data.astype(np.float32)
        return data

    bbox = gen_base(N)
    target = gen_base(N)
    mask = gen_base(N)
    return [bbox, target, mask]


register_sample(__name__, get_sample_config, gen_np_args)
