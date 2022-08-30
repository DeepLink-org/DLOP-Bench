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
        args_cases=[(892, 52, 32), (492, 26, 16), (292, 18, 4)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMSEG,
        url="https://github.com/open-mmlab/mmsegmentation/blob/504965184c3e6bc9ec43af54237129ef21981a5f/mmseg/ops/encoding.py#L34",  # noqa
        tags=[
            SampleTag.Reduce, SampleTag.ViewAttribute
        ],
    )


def gen_np_args(M, N, K):
    x = np.random.randn(2, M, N).astype(np.float32)
    codewords = np.random.randn(K, N).astype(np.float32)
    scale = -np.random.rand(K, ).astype(np.float32)
    return [x, codewords, scale]


register_sample(__name__, get_sample_config, gen_np_args)
