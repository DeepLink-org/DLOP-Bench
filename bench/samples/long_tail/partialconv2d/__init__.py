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
        args_cases=[(100, 3), (200, 3)],
        requires_grad=[False, False],
        backward=[False] * 2,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMEDIT,
        url="https://github.com/open-mmlab/mmediting/blob/ce2a70f1321907d1451a376637cf56be76168a67/mmedit/models/common/partial_conv.py#L10",  # noqa
        tags=[
            SampleTag.ThirdPartyCodes, SampleTag.IfElseBranch
        ],
    )


def gen_np_args(M, N):
    boxes = np.random.rand(M, N, 6, 6).astype(np.float32)
    mask = np.ones((M, N, 6, 6)).astype(np.float32)
    mask[..., 2, 2] = 0
    return [boxes, mask]


register_sample(__name__, get_sample_config, gen_np_args)
