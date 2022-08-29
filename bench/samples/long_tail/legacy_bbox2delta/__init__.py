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
        args_cases=[(3000, ), (2000, ), (1000, )],
        requires_grad=[False] * 4,
        backward=[False],
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/core/bbox/coder/legacy_delta_xywh_bbox_coder.py#L85",  # noqa
        tags=[SampleTag.ViewAttribute]
    )


def gen_np_args(N):
    def gen_base(row):
        data = np.random.rand(row, 4) * 100
        data = data.astype(np.float32)
        return data

    proposals = gen_base(N)
    gt = gen_base(N)
    means = (0.0, 0.0, 0.0, 0.0)
    stds = (1.0, 1.0, 1.0, 1.0)
    return [proposals, gt, means, stds]


register_sample(__name__, get_sample_config, gen_np_args)
