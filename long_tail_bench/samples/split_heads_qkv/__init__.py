from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(12, 12)],
        requires_grad=[False] * 6,
        backward=[False] * 3,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.PADDLEREC,
        url="https://github.com/PaddlePaddle/PaddleRec/blob/1f1d93eba5503bec87c11acc8203c2266c63c285/models/rank/bst/net.py#L339",  # noqa
        tags=[
            SampleTag.InputAware, SampleTag.ViewAttribute
        ],
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
