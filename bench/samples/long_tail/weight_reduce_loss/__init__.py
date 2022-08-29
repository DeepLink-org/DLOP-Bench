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
        args_cases=[(128, 4), (64, 4), (8, 4)],
        requires_grad=[True, False, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMSEG,
        url="https://github.com/open-mmlab/mmsegmentation/blob/504965184c3e6bc9ec43af54237129ef21981a5f/mmseg/models/losses/utils.py#L47",  # noqa
        tags=[
            SampleTag.Reduce, SampleTag.IfElseBranch
        ],
    )


def gen_np_args(M, N):
    loss_in = np.random.randn(M, N).astype(np.float32)
    weight_in = np.random.randn(M, N).astype(np.float32)
    return [loss_in, weight_in]


register_sample(__name__, get_sample_config, gen_np_args)
