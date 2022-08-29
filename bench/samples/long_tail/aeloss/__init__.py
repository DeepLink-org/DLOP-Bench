# Copyright (c) OpenComputeLab. All Rights Reserved.
from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(4, 12, 12), (4, 16, 16), (4, 8, 8)],
        requires_grad=[False] * 2,
        backward=[False] * 2,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMPOSE,
        url="https://github.com/open-mmlab/mmpose/blob/c0d5e4c2b39a1ebeca4be2c6c6358c33a9a5012d/mmpose/models/losses/multi_loss_factory.py#L68",  # noqa
        tags=[SampleTag.ForLoop, SampleTag.ViewAttribute, SampleTag.Reduce],
    )


def gen_np_args(K, W, H):
    shape1 = (K, W, H, 1)
    pred_tag = np.random.randint(0, 5, shape1)
    pred_tag = pred_tag.astype(np.float32)
    shape2 = (4, K, 2)
    joints = np.random.randint(0, 1, shape2)
    joints = joints.astype(np.float32)
    return [pred_tag, joints]


register_sample(__name__, get_sample_config, gen_np_args)
