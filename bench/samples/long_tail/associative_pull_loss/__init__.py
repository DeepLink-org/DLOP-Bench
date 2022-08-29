from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(0, 2, 3000, 4), (0, 3, 2000, 2), (0, 3, 6000, 8)],
        requires_grad=[False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[
            SampleTag.Reduce,
            SampleTag.ViewAttribute,
            SampleTag.AdvancedIndexing,
        ],
    )


def gen_np_args(M, N, K, J):
    boxes1 = np.random.randn(K, J)
    boxes1 = boxes1.astype(np.float32)

    boxes2 = np.random.randn(K, J)
    boxes2 = boxes2.astype(np.float32)

    mask = np.random.randint(M, N, (K, J))
    mask = mask.astype(np.float32)

    return [boxes1, boxes2, mask]


register_sample(__name__, get_sample_config, gen_np_args)
