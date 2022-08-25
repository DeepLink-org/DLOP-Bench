# Copyright(c) OpenMMLab. All Rights Reserved.
# Copied from
from bench.common import (
    SampleConfig,
    register_sample,
    SampleTag,
    SampleSource,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(4, 2048, 24, 37)],
        requires_grad=[False],
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/14602a71482082746399dd6bc5712a9aa8f804d2/models/position_encoding.py#L51",  # noqa
        tags=[SampleTag.Customized, SampleTag.ViewAttribute],
    )


def gen_np_args(M, N, K, Q):
    shape1 = (M, N, K, Q)

    x = np.ones(shape1)
    x = x.astype(np.float32)

    mask = np.random.randn(M, K, Q)
    mask = mask.astype(np.float32)

    return [x, mask]


register_sample(__name__, get_sample_config, gen_np_args)
