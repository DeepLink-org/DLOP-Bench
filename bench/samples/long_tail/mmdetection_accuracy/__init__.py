from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(32, 1000), (16, 2000), (8, 3000)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/losses/accuracy.py#L7",  # noqa
        tags=[
            SampleTag.ForLoop, SampleTag.IfElseBranch, SampleTag.ViewAttribute
        ])


def gen_np_args(N, M):
    output = np.random.rand(N, M).astype(np.float32)
    target = np.random.randint(0, 1000, size=(N, ), dtype=np.int64)
    return [output, target]


register_sample(__name__, get_sample_config, gen_np_args)
