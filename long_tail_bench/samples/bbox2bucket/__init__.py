from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000,), (2000,), (1000,)],
        requires_grad=[False] * 4,
        performance_iters=1000,
        backward=[False] * 4,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/core/bbox/coder/bucketing_bbox_coder.py#L145",  # noqa
        tags=[
            SampleTag.IfElseBranch,
            SampleTag.ForLoop,
            SampleTag.AdvancedIndexing,
            SampleTag.ViewAttribute,
        ],
    )


def gen_np_args(M):
    proposals = np.random.randn(M, 4)
    proposals = proposals.astype(np.float32)
    gt = np.random.randn(M, 4)
    gt = gt.astype(np.float32)
    return [proposals, gt]


register_sample(__name__, get_sample_config, gen_np_args)
