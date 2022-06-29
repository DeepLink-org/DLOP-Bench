from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(12, 1)],
        requires_grad=[False] * 2,
        backward=[False] * 2,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.PADDLEREC,
        url="https://github.com/PaddlePaddle/PaddleRec/blob/0aabcd30d5678d05181e6edea47c7948ef7baf12/models/rank/fm/net.py#L21",  # noqa
        tags=[SampleTag.InputAware, SampleTag.Reduce, SampleTag.Broadcast],
    )


def gen_np_args(N, W):
    shape = (N, W)
    input0 = np.random.randint(0, 5, shape)
    input0 = input0.astype(np.float32)
    input1 = np.random.randint(0, 5, shape)
    input1 = input1.astype(np.float32)
    return [input0, input1]


register_sample(__name__, get_sample_config, gen_np_args)
