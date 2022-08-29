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
        args_cases=[(32, 4), (16, 4), (64, 4)],
        requires_grad=[False],
        backward=[True, False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMACTION2,
        url="https://github.com/open-mmlab/mmaction2/blob/48ba0fb6dbceb0084f95f4ec288b23b244d2c360/mmaction/models/skeleton_gcn/base.py#L65",  # noqa
        tags=[
            SampleTag.ForLoop, SampleTag.IfElseBranch, SampleTag.Reduce,
            SampleTag.AdvancedIndexing
        ])


def gen_np_args(N, M):
    top1_acc = np.random.rand(N, M).astype(np.float32)
    top5_acc = np.random.rand(N, M).astype(np.float32)
    loss_cls = np.random.rand(N, M).astype(np.float32)
    return [top1_acc, top5_acc, loss_cls]


register_sample(__name__, get_sample_config, gen_np_args)
