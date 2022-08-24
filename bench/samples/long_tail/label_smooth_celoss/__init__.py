from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, 4), (2000, 2), (1000, 4)],
        requires_grad=[True, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[SampleTag.ViewAttribute, SampleTag.Reduce]
    )


def gen_np_args(M, N):
    def gen_boxes(row, column):
        data = np.random.randn(row, column)
        data = data.astype(np.float32)
        return data

    def gen_label(row):
        data = np.random.randint(0, 2, (row, 1))
        data = data.astype(np.float32)
        return data

    boxes = gen_boxes(M, N)
    label = gen_label(M)

    return [boxes, label]


register_sample(__name__, get_sample_config, gen_np_args)
