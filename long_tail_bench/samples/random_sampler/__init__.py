from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(4,)],
        requires_grad=[False] * 3,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/core/bbox/samplers/random_sampler.py#L9",  # noqa
        tags=[
            SampleTag.IfElseBranch,
            SampleTag.AdvancedIndexing,
            SampleTag.InputAware,
            SampleTag.ViewAttribute,
        ],
    )


def gen_np_args(N):
    boxes = np.random.randn(N, 4)
    boxes = boxes.astype(np.float32)

    gt_boxes = np.random.randn(N, 4)
    gt_boxes = gt_boxes.astype(np.float32)
    return [N, boxes, gt_boxes]


register_sample(__name__, get_sample_config, gen_np_args)
