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
        args_cases=[(10, ), (20, ), (30, )],
        requires_grad=[False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/dbolya/yolact/blob/57b8f2d95e62e2e649b382f516ab41f949b57239/layers/box_utils.py#L376",  # noqa
        tags=[
            SampleTag.ViewAttribute
        ],
    )


def gen_np_args(N):
    def gen_base(num):
        data = np.random.rand(num, num) * 10
        data = data.astype(np.int)
        return data

    src = gen_base(N)
    idx = gen_base(N)
    return [src, idx]


register_sample(__name__, get_sample_config, gen_np_args)
