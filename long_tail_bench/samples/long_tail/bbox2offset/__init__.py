from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, 4), (2000, 4), (1000, 4)],
        requires_grad=[False] * 3,
        rtol=1e-3,
        backward=[False],
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[SampleTag.IfElseBranch],
    )


def gen_np_args(M, N):
    boxes = np.random.randn(M, N)
    boxes = boxes.astype(np.float32)
    gt = np.random.randn(M, N)
    gt = gt.astype(np.float32)
    return [boxes, gt]


register_sample(__name__, get_sample_config, gen_np_args)
