from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(8, 8, 8, 3, 16), (12, 12, 12, 3, 16)],
        requires_grad=[False] * 6,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py#L10",  # noqa
        tags=[
            SampleTag.IfElseBranch,
            SampleTag.ForLoop,
            SampleTag.AdvancedIndexing,
            SampleTag.ViewAttribute,
        ],
    )


def gen_np_args(N, H, W, C, roi_num):
    num_levels = 5
    feats_list = []
    for _ in num_levels:
        feat = np.random.randn(N, H, W, C)
        feat = feat.astype(np.float32)
        feats_list.append(feat)

    rois = np.random.randn(roi_num, 4)
    rois = rois.astype(np.float32)
    return [feats_list, rois]


register_sample(__name__, get_sample_config, gen_np_args)
