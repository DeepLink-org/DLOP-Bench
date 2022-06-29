import numpy as np
from long_tail_bench.common import SampleConfig, register_sample
from long_tail_bench.core import registry
from tests.test_framework.one_sample.common import reset_count, count


def get_sample_config():
    return SampleConfig(
        args_cases=[(1, 1), (2, 3), (3, 3)],
        requires_grad=[True, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=True,
    )


def np_args_generator(M, N):
    a = np.full((M, N), 1, dtype=np.float32)
    b = np.full((M, N), 2, dtype=np.float32)
    return [a, b]


test_sample_name = str(__name__).split(".")[-1]
register_sample(__name__, get_sample_config, np_args_generator)

__all__ = ["test_sample_name", "count", "reset_count", "registry"]
