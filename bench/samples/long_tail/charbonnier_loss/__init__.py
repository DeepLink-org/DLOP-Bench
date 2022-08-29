# Copyright (c) OpenComputeLab. All Rights Reserved.
from bench.common import SampleConfig, register_sample, SampleSource
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(300, 8), (600, 16), (800, 32)],
        requires_grad=[True, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMEDIT,
        url="https://github.com/open-mmlab/mmediting/blob/ce2a70f1321907d1451a376637cf56be76168a67/mmedit/models/losses/pixelwise_loss.py#L41",  # noqa
    )


def gen_np_args(M, N):
    boxes = np.random.rand(M, N)
    gt = np.random.rand(M, N)

    return [boxes, gt]


register_sample(__name__, get_sample_config, gen_np_args)
