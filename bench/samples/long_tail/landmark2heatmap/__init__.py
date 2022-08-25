# Copyright(c) OpenMMLab. All Rights Reserved.
# Copied from
from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(30, 78), (4, 78), (16, 78)],
        requires_grad=[False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[SampleTag.ViewAttribute]
    )


def gen_np_args(M, N):
    def gen_base(row, column):
        data = np.random.randn(row, column) * 100
        data = data.astype(np.float32)
        return data

    boxes = gen_base(M, N)

    return [boxes]


register_sample(__name__, get_sample_config, gen_np_args)
