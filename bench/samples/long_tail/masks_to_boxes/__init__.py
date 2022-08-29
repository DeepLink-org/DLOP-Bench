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
        args_cases=[(400, 128, 6), (300, 128, 6)],
        requires_grad=[False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/util/box_ops.py#L64",  # noqa
        tags=[SampleTag.ViewAttribute, \
              SampleTag.IfElseBranch, SampleTag.Reduce]
    )


def gen_np_args(M, N, K):
    def gen_base(x, y, z):
        data = np.random.randn(x, y, z) * 100
        data = data.astype(np.float32)
        return data

    boxes = gen_base(M, N, K)

    return [boxes]


register_sample(__name__, get_sample_config, gen_np_args)
