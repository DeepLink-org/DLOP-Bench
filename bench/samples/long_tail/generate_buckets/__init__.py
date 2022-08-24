from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(64, 4), (128, 4), (32, 4)],
        requires_grad=[False] * 3,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/1a90fa80a761fe15e69111a625d82874ed783f7b/mmdet/core/bbox/coder/bucketing_bbox_coder.py#L96",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.IfElseBranch]
    )


def gen_np_args(M, N):
    proposals = np.random.rand(M, N)
    proposals = proposals.astype(np.float32)
    num_buckets = 8
    scale_factor = 1.0
    return [proposals, num_buckets, scale_factor]


register_sample(__name__, get_sample_config, gen_np_args)
