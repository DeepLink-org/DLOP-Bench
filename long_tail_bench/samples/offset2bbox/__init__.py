from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(4000, 4), (5000, 4), (6000, 4)],
        requires_grad=[False] * 3,
        backward=[False],
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[
            SampleTag.ViewAttribute, SampleTag.IfElseBranch,
            SampleTag.ThirdPartyCodes
        ],
    )


def gen_np_args(M, N):
    boxes = np.random.randn(M, N).astype(np.float32)
    offset = np.random.randn(M, N).astype(np.float32)
    weights = np.ones(4)
    return [boxes, offset, weights]


register_sample(__name__, get_sample_config, gen_np_args)
