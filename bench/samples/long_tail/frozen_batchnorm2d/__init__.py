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
        args_cases=[(4,), (8,), (12,)],
        requires_grad=[False],
        performance_iters=1000,
        backward=[False],
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/14602a71482082746399dd6bc5712a9aa8f804d2/models/backbone.py#L19",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(M):
    x = np.random.randn(M, 4, 4, 4)
    x = x.astype(np.float32)
    return [x]


register_sample(__name__, get_sample_config, gen_np_args)
