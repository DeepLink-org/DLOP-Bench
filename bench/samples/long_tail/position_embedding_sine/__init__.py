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
        args_cases=[(4,), (6,), (8,)],
        requires_grad=[False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/14602a71482082746399dd6bc5712a9aa8f804d2/models/position_encoding.py#L12",  # noqa
        tags=[
            SampleTag.Customized,
            SampleTag.Reduce,
            SampleTag.ViewAttribute,
            SampleTag.IfElseBranch,
        ],
    )


def gen_np_args(M):
    shape1 = (M, M, M, M)
    x = np.ones(shape1)
    x = x.astype(np.float32)

    mask = np.random.randn(M, M, M)
    mask = mask.astype(np.float32)
    return [x, mask]


register_sample(__name__, get_sample_config, gen_np_args)
