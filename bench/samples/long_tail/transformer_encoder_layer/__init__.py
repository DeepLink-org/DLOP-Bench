from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(8, 3, 8, 8)],
        requires_grad=[False] * 4,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/14602a71482082746399dd6bc5712a9aa8f804d2/models/transformer.py#L62",  # noqa
        tags=[SampleTag.IfElseBranch],
    )


def gen_np_args(N, C, H, W):
    src = np.random.randn(H * W, N, C)
    src = src.astype(np.float32)

    mask = np.random.randn(H * W, N, C)
    mask = mask.astype(np.float32)

    pos_embed = np.random.randn(H * W, N, C)
    pos_embed = pos_embed.astype(np.float32)
    return [src, mask, pos_embed]


register_sample(__name__, get_sample_config, gen_np_args)
