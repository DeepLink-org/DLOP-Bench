from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(400, 72), (800, 144), (1600, 288)],
        requires_grad=[False] * 2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/util/box_ops.py#L40",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.Reduce],
    )


def gen_np_args(M, N):
    boxes1 = np.random.rand(M, 4)
    boxes1 = boxes1.astype(np.float32)
    boxes1[:, 2:] += 1
    boxes2 = np.random.rand(N, 4)
    boxes2 = boxes2.astype(np.float32)
    boxes2[:, 2:] += 1
    return [boxes1, boxes2]


register_sample(__name__, get_sample_config, gen_np_args)
