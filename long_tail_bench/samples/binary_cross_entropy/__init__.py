from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(100,), (200,), (300,)],
        requires_grad=[True, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/68d860d6f1a4a1fb687f2512038f2cbce7c6b08d/mmdet/models/losses/cross_entropy_loss.py#L58",  # noqa
        tags=[
            SampleTag.IfElseBranch,
            SampleTag.InputAware,
            SampleTag.AdvancedIndexing,
            SampleTag.ViewAttribute,
            SampleTag.Reduce,
        ],
    )


def gen_np_args(N):
    logit = np.random.rand(N, 1).astype(np.float32)
    label = np.random.randint(5, size=(N, )).astype(np.float32)
    return [logit, label]


register_sample(__name__, get_sample_config, gen_np_args)
