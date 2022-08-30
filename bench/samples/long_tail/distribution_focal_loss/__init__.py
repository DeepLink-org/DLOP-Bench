# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import SampleConfig, register_sample, SampleSource
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(128, 4), (256, 4), (512, 4)],
        requires_grad=[True, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/models/losses/gfocal_loss.py#L57"  # noqa
    )


def gen_np_args(M, N):
    pred = np.random.randn(M, N).astype(np.float32)
    target = np.random.randint(0, 2, (M, )).astype(np.float32)
    return [pred, target]


register_sample(__name__, get_sample_config, gen_np_args)
