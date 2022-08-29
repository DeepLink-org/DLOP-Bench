from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(12, 12, 12), (16, 16, 16), (32, 32, 32)],
        requires_grad=[False] * 1,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.PADDLEREC,
        url="https://github.com/PaddlePaddle/PaddleRec/blob/3b416accde4366e26f55f8d27f2453704a146bad/models/multitask/dselect_k/net.py#L121",  # noqa
        tags=[
            SampleTag.Broadcast
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
