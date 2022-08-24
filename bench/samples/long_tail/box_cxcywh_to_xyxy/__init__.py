from bench.common import SampleConfig, register_sample, SampleSource
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(50, 4), (100, 4), (200, 4)],
        requires_grad=[False] * 1,
        performance_iters=1000,
        backward=[False],
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/facebookresearch/detr/blob/eb9f7e03ed8e2ed2cd55528989fe7df890bc3fc0/util/box_ops.py#L9"  # noqa
    )


def gen_np_args(M, N):
    x = np.random.randn(M, N)
    x = x.astype(np.float32)
    return [x]


register_sample(__name__, get_sample_config, gen_np_args)
