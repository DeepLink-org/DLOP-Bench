# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import SampleConfig, register_sample, SampleSource
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(128, 4), (256, 8), (512, 16)],
        requires_grad=[False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/models/losses/balanced_l1_loss.py#L13",  # noqa
    )


def gen_np_args(M, N):
    pred = np.random.randn(M, N)
    pred = pred.astype(np.float32)
    target = np.random.randn(M, N)
    target = target.astype(np.float32)
    return [pred, target]


register_sample(__name__, get_sample_config, gen_np_args)
