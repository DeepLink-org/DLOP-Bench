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
        args_cases=[(16, 16)],
        requires_grad=[False] * 3,
        backward=[False] * 2,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.PADDLEREC,
        url="https://github.com/PaddlePaddle/PaddleRec/blob/fa3e3a8a591d55778edf8d26982467d5be0840ef/models/recall/mhcn/net.py#L224",  # noqa
        tags=[SampleTag.InputAware, SampleTag.Reduce, SampleTag.ForLoop],
    )


def gen_np_args(N, W):
    shape = (N, W)
    input0 = np.random.randint(0, 5, shape)
    input0 = input0.astype(np.float32)
    input1 = np.random.randint(0, 5, shape)
    input1 = input1.astype(np.float32)
    input2 = np.random.randint(0, 5, shape)
    input2 = input2.astype(np.float32)
    return [input0, input1, input2]


register_sample(__name__, get_sample_config, gen_np_args)
