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
        args_cases=[(8, 10, 16)],
        requires_grad=[False] * 2,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/14602a71482082746399dd6bc5712a9aa8f804d2/models/matcher.py#L12",  # noqa
        tags=[
            SampleTag.ViewAttribute,
            SampleTag.BuiltInDataStructure,
            SampleTag.AdvancedIndexing,
            SampleTag.ThirdPartyCodes,
        ],
    )


def gen_np_args(batch_size, num_queries, num_target_boxes):
    num_classes = 3
    pred_logits = np.random.randn(batch_size, num_queries, num_classes)
    pred_logits = pred_logits.astype(np.float32)
    pred_boxes = np.random.randn(batch_size, num_queries, 4)
    pred_boxes = pred_boxes.astype(np.float32)

    targets_list = []
    for _ in range(batch_size):
        labels = np.random.randn(num_target_boxes)
        labels = pred_logits.astype(np.float32)
        boxes = np.random.randn(num_target_boxes, 4)
        boxes = boxes.astype(np.float32)
        targets_list.append([labels, boxes])
    return [pred_logits, pred_boxes, targets_list]


register_sample(__name__, get_sample_config, gen_np_args)
