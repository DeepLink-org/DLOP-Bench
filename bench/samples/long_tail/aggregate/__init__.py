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
        args_cases=[(8, 12), (4, 8), (8, 16)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMSEG,
        url="https://github.com/open-mmlab/mmsegmentation/blob/504965184c3e6bc9ec43af54237129ef21981a5f/mmseg/ops/encoding.py#L47",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.Reduce],
    )


def gen_np_args(M, N):
    assigment_weights = np.random.randn(2, 8, M)
    assigment_weights = assigment_weights.astype(np.float32)

    x = np.random.randn(2, 8, 1)
    x = x.astype(np.float32)

    codewords = np.random.randn(M, N)
    codewords = codewords.astype(np.float32)
    return [assigment_weights, x, codewords]


register_sample(__name__, get_sample_config, gen_np_args)
