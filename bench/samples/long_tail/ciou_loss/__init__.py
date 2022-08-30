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
        args_cases=[(1000, ), (2000, ), (3000, )],
        requires_grad=[False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/models/losses/iou_loss.py#L177",  # noqa
        tags=[SampleTag.Reduce]
    )


def gen_np_args(M):
    pred = np.random.randn(M, 4)
    target = np.random.randn(M, 4)

    return [pred, target]


register_sample(__name__, get_sample_config, gen_np_args)
