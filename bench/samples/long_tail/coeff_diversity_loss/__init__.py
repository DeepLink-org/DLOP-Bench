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
        args_cases=[(300, ), (400, ), (600, )],
        requires_grad=[False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/dbolya/yolact/blob/57b8f2d95e62e2e649b382f516ab41f949b57239/layers/modules/multibox_loss.py#L475",  # noqa
        tags=[
            SampleTag.ViewAttribute
        ],
    )


def gen_np_args(N):
    def gen_base(num):
        data = np.random.randn(num, 3)
        data = data.astype(np.float32)
        return data

    coeffs = gen_base(N)
    instance_t = np.random.randn(N, 1)
    instance_t = instance_t.astype(np.float32)
    return [coeffs, instance_t]


register_sample(__name__, get_sample_config, gen_np_args)
