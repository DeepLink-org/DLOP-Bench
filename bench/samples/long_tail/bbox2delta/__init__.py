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
        args_cases=[(128,), (64,), (8,)],
        requires_grad=[False] * 4,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L117",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(N):
    def gen_base(num):
        data = np.random.randn(num, 4) * 100
        data = data.astype(np.float32)
        return data

    proposals = gen_base(N)
    gt = gen_base(N)
    means = np.zeros(4)
    stds = np.ones(4)
    return [proposals, gt, means, stds]


register_sample(__name__, get_sample_config, gen_np_args)
