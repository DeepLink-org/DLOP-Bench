from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(32, 1000), (16, 500), (32, 500)],
        requires_grad=[False, False, False],
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.PARROTS_EXAMPLE,
        url="https://gitlab.bj.sensetime.com/yangbo1/parrots.example/-/blob/master/benchmark/parallel/benchmark_linklink.py#L417",  # noqa
        tags=[SampleTag.Reduce, SampleTag.ViewAttribute, SampleTag.ForLoop]
    )


def gen_np_args(M, N):
    output = np.random.randn(M, N)
    output = output.astype(np.float32)
    target = np.random.randint(0, 1000, (M, ))
    target = target.astype(np.int64)
    return [output, target]


register_sample(__name__, get_sample_config, gen_np_args)
