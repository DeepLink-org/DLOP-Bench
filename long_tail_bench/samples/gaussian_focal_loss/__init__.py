from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(128, 4), (256, 4), (512, 4)],
        requires_grad=[False, False, False, False],
        backward=[False],
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/bde7b4b7eea9dd6ee91a486c6996b2d68662366d/mmdet/models/losses/gaussian_focal_loss.py#L11",  # noqa
        tags=[],
    )


def gen_np_args(M, N):
    pred = np.random.rand(M, N)
    pred = pred.astype(np.float32)
    target = np.random.randint(0, 2, (M, N))
    target = target.astype(np.float32)
    return [pred, target, 2.0, 4.0]


register_sample(__name__, get_sample_config, gen_np_args)
