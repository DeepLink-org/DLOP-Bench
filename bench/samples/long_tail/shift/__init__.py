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
        args_cases=[
            (64, 64, 56, 56, 8),
            (64, 128, 56, 56, 8),
            (64, 256, 56, 56, 8),
        ],
        requires_grad=[True, False, False],
        backward=[True],
        performance_iters=50,
        save_timeline=False,
        source=SampleSource.MMACTION2,
        url="https://github.com/open-mmlab/mmaction2/blob/48ba0fb6dbceb0084f95f4ec288b23b244d2c360/mmaction/models/backbones/resnet_tsm.py#L73",  # noqa
        tags=[
            SampleTag.AdvancedIndexing, SampleTag.ViewAttribute
        ],
    )


def gen_np_args(M, N, K, Q, J):
    num_segments = J
    shift_div = J
    x = np.random.rand(M, N, K, Q).astype(np.float32)
    return [x, num_segments, shift_div]


register_sample(__name__, get_sample_config, gen_np_args)
