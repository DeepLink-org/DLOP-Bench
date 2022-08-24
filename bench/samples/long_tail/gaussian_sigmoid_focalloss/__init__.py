from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(300, 4), (400, 4), (200, 4)],
        requires_grad=[True, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.POD,
        url="https://gitlab.bj.sensetime.com/platform/ParrotsDL/pytorch-object-detection/-/blob/master/pod/models/losses/focal_loss.py#L202",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.InputAware],
    )


def gen_np_args(M, N):
    input = np.random.randn(M, N)
    input = input.astype(np.float32)
    gt = np.random.randint(0, 2, (M, N))
    gt = gt.astype(np.float32)
    return [input, gt]


register_sample(__name__, get_sample_config, gen_np_args)
