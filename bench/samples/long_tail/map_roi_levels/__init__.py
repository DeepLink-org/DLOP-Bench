from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(20, 5), (40, 8), (60, 12)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py#L36",  # noqa
        tags=[SampleTag.ViewAttribute]
    )


def gen_np_args(M, N):
    def gen_base(row, column):
        data = np.zeros((row, column))
        data = data.astype(np.float32)
        data[:, 3] = 1
        return data

    rois = gen_base(M, N)

    num_levels = 4
    finest_scale = 56

    return [rois, num_levels, finest_scale]


register_sample(__name__, get_sample_config, gen_np_args)
