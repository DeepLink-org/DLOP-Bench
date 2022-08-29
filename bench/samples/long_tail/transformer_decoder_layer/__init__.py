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
        args_cases=[(8, 3, 8, 8)],
        requires_grad=[False] * 5,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/14602a71482082746399dd6bc5712a9aa8f804d2/models/transformer.py#L86",  # noqa
        tags=[SampleTag.IfElseBranch],
    )


def gen_np_args(N, C, H, W):
    pos_embed = np.random.randn(H * W, N, C)
    pos_embed = pos_embed.astype(np.float32)

    query_embed = np.random.randn(H * W, N, C)
    query_embed = query_embed.astype(np.float32)
    tgt = np.zeros_like(query_embed)

    memory = np.random.randn(H * W, N, C)
    memory = memory.astype(np.float32)

    mask = np.random.randn(H * W, N, C)
    mask = mask.astype(np.float32)

    return [tgt, memory, mask, pos_embed, query_embed]


register_sample(__name__, get_sample_config, gen_np_args)
