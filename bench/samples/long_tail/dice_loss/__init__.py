from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3, 4, 5, 6), (4, 5, 6, 7), (5, 6, 7, 8)],
        requires_grad=[True, False],
        backward=[False],
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[SampleTag.IfElseBranch, SampleTag.Reduce,
              SampleTag.ViewAttribute]
    )


def gen_np_args(M, N, P, Q):
    boxes = np.random.randn(M, N, P, Q)
    boxes = boxes.astype(np.float32)
    gt = np.random.randn(M, N, P, Q)
    gt = gt.astype(np.float32)
    return [boxes, gt]


register_sample(__name__, get_sample_config, gen_np_args)
