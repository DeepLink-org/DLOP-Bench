from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(4, 4, 10), (8, 4, 10), (12, 4, 10)],
        requires_grad=[False] * 5,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        tags=[
            SampleTag.InputAware,
            SampleTag.ViewAttribute,
            SampleTag.IfElseBranch,
        ],
    )


def gen_np_args(M, N, K):
    bbox_targets = np.random.randn(M, N)
    bbox_targets = bbox_targets.astype(np.float32)
    bbox_weights = np.random.randn(M, N)
    bbox_weights = bbox_weights.astype(np.float32)
    labels = np.ones((M,), np.int64)
    bbox_targets_expand = np.zeros((M, N), np.float32)
    bbox_weights_expand = np.zeros((M, N), np.float32)

    return [
        bbox_targets,
        bbox_weights,
        labels,
        bbox_targets_expand,
        bbox_weights_expand,
    ]


register_sample(__name__, get_sample_config, gen_np_args)
