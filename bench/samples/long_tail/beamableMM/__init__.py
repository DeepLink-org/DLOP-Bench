from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(8, 12, 12), (16, 16, 16), (32, 8, 8)],
        requires_grad=[False] * 2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.FAIRSEQ,
        url="https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/modules/beamable_mm.py#L10",  # noqa
        tags=[
            SampleTag.IfElseBranch, SampleTag.ViewAttribute,
            SampleTag.AdvancedIndexing
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
