from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(4, 16), (8, 8), (16, 4)],
        requires_grad=[False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/core/bbox/iou_calculators/iou2d_calculator.py#L75",  # noqa
        tags=[SampleTag.IfElseBranch],
    )


def gen_np_args(M, N):
    bboxes1 = np.random.randn(32, M, 4)
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = np.random.randn(32, N, 4)
    bboxes2 = bboxes2.astype(np.float32)
    pred = np.random.randn(M, 4) * 100
    pred = pred.astype(np.float32)
    target = np.random.randn(M, 4) * 100
    target = target.astype(np.float32)
    return [bboxes1, bboxes2, pred, target]


register_sample(__name__, get_sample_config, gen_np_args)
