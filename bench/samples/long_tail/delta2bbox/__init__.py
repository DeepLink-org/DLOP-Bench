from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, 4), (2000, 4), (1000, 4)],
        requires_grad=[False] * 5,
        rtol=1e-5,
        atol=1e-5,
        backward=[False],
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L164",  # noqa
        tags=[SampleTag.ThirdPartyCodes, SampleTag.IfElseBranch,
              SampleTag.ViewAttribute]
    )


def gen_np_args(M, N):
    rois = np.random.randn(M, N)
    deltas = np.random.randn(M, N)

    means = np.zeros(N, )
    stds = np.ones(N, )
    max_shape = np.random.randn(3)

    return [rois, deltas, means, stds, max_shape]


register_sample(__name__, get_sample_config, gen_np_args)
