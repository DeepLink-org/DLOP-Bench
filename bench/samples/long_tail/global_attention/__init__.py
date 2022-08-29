from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(1, 6, 3)],
        requires_grad=[False] * 2,
        backward=[False] * 2,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.OEPNNMT,
        url="https://github.com/OpenNMT/OpenNMT-py/blob/7c314f41dc1b017ac105144beeb53cb072960a54/onmt/modules/global_attention.py#L15",  # noqa
        tags=[
            SampleTag.IfElseBranch, SampleTag.InputAware,
            SampleTag.ViewAttribute
        ],
    )


def gen_np_args(N, W, H):
    shape = (N, W, H)
    input1 = np.random.randint(0, 5, shape)
    input1 = input1.astype(np.float32)
    input2 = np.random.randint(0, 1, shape)
    input2 = input2.astype(np.float32)
    return [input1, input2]


register_sample(__name__, get_sample_config, gen_np_args)
