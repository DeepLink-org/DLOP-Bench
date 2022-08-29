# Copyright (c) OpenComputeLab. All Rights Reserved.
from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np

# crop results:
# parrots: s1 ~ s3 pass


def get_sample_config():
    return SampleConfig(
        args_cases=[(4, ), (2, ), (8, )],
        requires_grad=[False, False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/dbolya/yolact/blob/57b8f2d95e62e2e649b382f516ab41f949b57239/layers/box_utils.py#L350",  # noqa
        tags=[
            SampleTag.Reduce, SampleTag.IfElseBranch
        ],
    )


def gen_np_args(N):
    def gen_boxes(num):
        data = np.random.randn(num, 4) * 100
        data = data.astype(np.float32)
        return data

    def gen_masks(num):
        data = np.random.randn(num, num, num) * 100
        data = data.astype(np.float32)
        return data

    masks = gen_masks(N)
    boxes = gen_boxes(N)
    padding = 1
    return [masks, boxes, padding]


register_sample(__name__, get_sample_config, gen_np_args)
