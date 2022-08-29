# Copyright (c) OpenComputeLab. All Rights Reserved.
from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np

# sanitize_coordinates
# result: s1 ~ s4 pass


def get_sample_config():
    return SampleConfig(
        args_cases=[(300, 100), (600, 100), (800, 100)],
        requires_grad=[False] * 5,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/models/dense_heads/yolact_head.py#L876",  # noqa
        tags=[
            SampleTag.Reduce, SampleTag.IfElseBranch
        ],
    )


def gen_np_args(N, M):
    def gen_base(row, column):
        data = np.random.randint(0, 100, size=(row, column), dtype=np.int64)
        data = data.astype(np.float32)
        return data

    x1 = gen_base(N, M)
    x2 = gen_base(N, M)
    return [x1, x2]


register_sample(__name__, get_sample_config, gen_np_args)
