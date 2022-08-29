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
        args_cases=[(3000, ), (2000, ), (1000, )],
        requires_grad=[False] * 5,
        performance_iters=1000,
        backward=[False] * 2,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/1a90fa80a761fe15e69111a625d82874ed783f7b/mmdet/core/bbox/coder/bucketing_bbox_coder.py#L269",  # noqa
        tags=[SampleTag.ThirdPartyCodes, SampleTag.ViewAttribute,
              SampleTag.IfElseBranch, SampleTag.Reduce,
              SampleTag.AdvancedIndexing]
    )


def gen_np_args(M):
    proposals = np.random.rand(M, 4)
    cls_preds = np.random.rand(M, 16)
    offset_preds = np.random.rand(M, 16)
    num_buckets = 8
    scale_factor = 1.0
    return [proposals, cls_preds, offset_preds, num_buckets, scale_factor]


register_sample(__name__, get_sample_config, gen_np_args)
