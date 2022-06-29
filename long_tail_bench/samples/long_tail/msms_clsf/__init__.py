from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(128, 4), (256, 4), (512, 4)],
        requires_grad=[True, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[
            SampleTag.AdvancedIndexing, SampleTag.IfElseBranch,
            SampleTag.Reduce
        ],
    )


def gen_np_args(M, N):
    predict = np.random.randn(M, N)
    label = np.random.randint(0, 2, (M, ), dtype=np.int64)

    return [predict, label]


register_sample(__name__, get_sample_config, gen_np_args)
