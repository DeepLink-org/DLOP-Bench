# Copyright (c) OpenComputeLab. All Rights Reserved.
from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)


def get_sample_config():
    return SampleConfig(
        args_cases=[(160, 120, 3), (320, 240, 6), (480, 360, 9)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/core/anchor/point_generator.py#L30",  # noqa
        tags=[
            SampleTag.AdvancedIndexing, SampleTag.ViewAttribute
        ],
    )


def gen_np_args(M, N, K):
    featmap_size = (M, N)
    valid_size = (M, N)
    num_base_anchors = K
    return [featmap_size, valid_size, num_base_anchors]


register_sample(__name__, get_sample_config, gen_np_args)
