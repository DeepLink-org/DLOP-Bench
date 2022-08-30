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
        args_cases=[(8, ), (16, ), (32, )],
        requires_grad=[False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.POD,
        url="https://gitlab.bj.sensetime.com/liuyan/pytorch-object-detection/-/blob/master/pod/models/heads/centernetkp_head/centernetkp_head.py#L80",  # noqa
        tags=[SampleTag.ForLoop, SampleTag.ViewAttribute,
              SampleTag.AdvancedIndexing, SampleTag.Reduce,
              SampleTag.BuiltInDataStructure]
    )


def gen_np_args(N):
    heatmap1 = np.random.randn(N, 3, 8, 8)
    embedding1 = np.random.randn(N, 1, 8, 8)
    offset1 = np.random.randn(N, 2, 8, 8)

    heatmap2 = np.random.randn(N, 3, 8, 8)
    embedding2 = np.random.randn(1, 1, 2, 8, 8)
    B = 1
    offset2 = np.random.randn(N, B, 2, 8, 8)

    return [heatmap1, embedding1, offset1, heatmap2, embedding2, offset2, N]


register_sample(__name__, get_sample_config, gen_np_args)
