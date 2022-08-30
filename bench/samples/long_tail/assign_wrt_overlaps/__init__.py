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
        args_cases=[(4, 4), ],
        requires_grad=[False, False],
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/core/bbox/assigners/max_iou_assigner.py#L128",  # noqa
        tags=[
            SampleTag.IfElseBranch,
            SampleTag.AdvancedIndexing,
            SampleTag.ForLoop,
            SampleTag.InputAware,
            SampleTag.Customized
        ],
    )


def gen_np_args(K, N):
    overlaps = np.random.rand(K, N)
    overlaps = overlaps.astype(np.float32)
    return [overlaps]


register_sample(__name__, get_sample_config, gen_np_args)
