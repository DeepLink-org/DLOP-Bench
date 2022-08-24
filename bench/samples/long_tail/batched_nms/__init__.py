from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(32,), (64,), (128,)],
        requires_grad=[False] * 4,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMCV,
        url="https://github.com/open-mmlab/mmcv/blob/add157cc73b05c7be3d9b63505e17dc2288a6d0f/mmcv/ops/nms.py#L260",  # noqa
        tags=[
            SampleTag.BuiltInDataStructure,
            SampleTag.IfElseBranch,
            SampleTag.InputAware,
            SampleTag.ViewAttribute,
            SampleTag.AdvancedIndexing,
        ],
    )


def gen_np_args(N):
    boxes = np.random.randn(N, 4)
    boxes = boxes.astype(np.float32)
    scores = np.random.randn(N)
    scores = scores.astype(np.float32)
    idxs = np.random.randint(0, N, (N,))
    idxs = idxs.astype(np.float32)
    return [boxes, scores, idxs]


register_sample(__name__, get_sample_config, gen_np_args)
