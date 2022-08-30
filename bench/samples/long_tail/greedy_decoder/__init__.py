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
        args_cases=[(12, 12, 12), (16, 16, 16), (32, 32, 32)],
        requires_grad=[False] * 3,
        backward=[False] * 2,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.ASSEMBLYAI,
        url="https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO#scrollTo=RVJs4Bk8FjjO",  # noqa
        tags=[
            SampleTag.ForLoop, SampleTag.Reduce, SampleTag.AdvancedIndexing,
            SampleTag.BuiltInDataStructure
        ],
    )


def gen_np_args(N, W, H):
    shape = (N, W, H)
    input1 = np.random.randint(0, 5, shape)
    input1 = input1.astype(np.float32)
    input2 = np.random.randint(0, 1, (N, N))
    input2 = input2.astype(np.float32)
    return [input1, input2]


register_sample(__name__, get_sample_config, gen_np_args)
