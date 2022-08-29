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
        args_cases=[(42, 470, 512), (32, 450, 512)],
        requires_grad=[False] * 9,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/f08548bfd6d394a82566022709b5ce9e6b0a855e/mmdet/models/roi_heads/bbox_heads/bbox_head.py#L122",  # noqa
        tags=[SampleTag.IfElseBranch],
    )


def gen_np_args(N, M, Q):
    def gen_base(num):
        data = np.random.randn(num, 4) * 100
        data = data.astype(np.float32)
        return data

    pos_bboxes = gen_base(N)
    neg_bboxes = gen_base(M)
    pos_gt_bboxes = gen_base(N)
    pos_gt_labels = np.random.randint(0, 5, (N, )).astype(np.float32)
    labels = np.random.randint(0, 5, (Q, )).astype(np.float32)
    label_weights = np.random.randn(Q, 4).astype(np.float32)
    bbox_targets = gen_base(Q)
    bbox_weights = gen_base(Q)
    return [
        pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, labels,
        label_weights, bbox_targets, bbox_weights
    ]


register_sample(__name__, get_sample_config, gen_np_args)
