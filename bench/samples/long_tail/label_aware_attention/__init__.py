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
        args_cases=[(12, 12, 12), (16, 16, 16), (32, 32, 32)],
        requires_grad=[False] * 2,
        backward=[False] * 2,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.PADDLEREC,
        url="https://github.com/PaddlePaddle/PaddleRec/blob/8ce9dbdbc5d0149bfcdb57a06a183024cff21aa3/models/recall/mind/net.py#L267",  # noqa
        tags=[
            SampleTag.Reduce
        ],
    )


def gen_np_args(N, W, H):
    shape = (N, N)
    input1 = np.random.randint(0, 5, shape)
    input1 = input1.astype(np.float32)
    input2 = np.random.randint(0, 5, shape)
    input2 = input1.astype(np.float32)
    return [input1, input2]


register_sample(__name__, get_sample_config, gen_np_args)
