from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(24, 1)],
        requires_grad=[False] * 1,
        backward=[False] * 2,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.PADDLEREC,
        url="https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/utils.py#L256",  # noqa
        tags=[SampleTag.InputAware, SampleTag.Broadcast],
    )


def gen_np_args(N, W):
    shape = (N, W)
    input1 = np.random.randint(0, 5, shape)
    input1 = input1.astype(np.float32)
    input2 = np.random.randint(0, 5, shape)
    input2 = input1.astype(np.float32)
    input3 = np.random.randint(0, 5, shape)
    input3 = input1.astype(np.float32)
    return [input1, input2, input3]


register_sample(__name__, get_sample_config, gen_np_args)
