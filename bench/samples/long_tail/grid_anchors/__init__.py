from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(16, 32, 8), (32, 16, 8), (16, 16, 8)],
        requires_grad=[False] * 3,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/4bd4ace53a626659070204236da5b8e14ae9cd08/mmdet/core/anchor/anchor_generator.py#L318",  # noqa
        tags=[SampleTag.ViewAttribute]
    )


def gen_np_args(M, N, K):
    def gen_base(row):
        data = np.random.randn(row, 4)
        data = data.astype(np.float32)
        return data

    base_anchors = gen_base(M)
    featmap_size = (M, N)
    stride = K

    return [base_anchors, featmap_size, stride]


register_sample(__name__, get_sample_config, gen_np_args)
