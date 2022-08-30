# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleTag,
    SampleSource,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(8, 10)],
        requires_grad=[False] * 2,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/14602a71482082746399dd6bc5712a9aa8f804d2/models/detr.py#L258",  # noqa
        tags=[SampleTag.BuiltInDataStructure, SampleTag.ViewAttribute],
    )


def gen_np_args(batch_size, num_queries):
    num_classes = 3
    pred_logits = np.random.randn(batch_size, num_queries, num_classes)
    pred_logits = pred_logits.astype(np.float32)
    pred_boxes = np.random.randn(batch_size, num_queries, 4)
    pred_boxes = pred_boxes.astype(np.float32)
    target_sizes = np.random.randint(1, 50, (batch_size, 2))
    target_sizes = target_sizes.astype(np.float32)
    return [pred_logits, pred_boxes, target_sizes]


register_sample(__name__, get_sample_config, gen_np_args)
