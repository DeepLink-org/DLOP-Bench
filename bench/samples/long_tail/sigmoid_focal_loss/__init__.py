from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(3000, 4), (4000, 4), (5000, 4)],
        requires_grad=[True, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMCLS,
        url="https://github.com/open-mmlab/mmclassification/blob/72cffac7b50a556824907721665e6705fd5045fb/mmcls/models/losses/focal_loss.py#L9",  # noqa
        tags=[
            SampleTag.IfElseBranch, SampleTag.ViewAttribute
        ],
    )


def gen_np_args(M, N):
    pred = np.random.randn(M, N).astype(np.float32)
    target = np.random.randn(M, N).astype(np.float32)
    return [pred, target]


register_sample(__name__, get_sample_config, gen_np_args)
