from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(400, 4, 72, 4), (800, 4, 144, 4), (1600, 4, 288, 4)],
        requires_grad=[False] * 2,
        backward=[False] * 2,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/util/box_ops.py#L24",  # noqa
        tags=[SampleTag.Reduce]
    )


def gen_np_args(M1, N1, M2, N2):
    boxes1 = np.ones((M1, N1), np.float32)
    boxes2 = np.ones((M2, N2), np.float32)
    return [boxes1, boxes2]


register_sample(__name__, get_sample_config, gen_np_args)
