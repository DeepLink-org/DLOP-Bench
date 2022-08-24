from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(4, 3, 1), (8, 6, 2), (16, 12, 4)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/4bd4ace53a626659070204236da5b8e14ae9cd08/mmdet/core/anchor/anchor_generator.py#L131",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(M, N, K):
    base_size = M
    ratios = np.random.randn(N)
    ratios = ratios.astype(np.float32)
    scales = np.random.randn(K)
    scales = scales.astype(np.float32)
    return [base_size, ratios, scales]


register_sample(__name__, get_sample_config, gen_np_args)
