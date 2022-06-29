from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(128, 4), (256, 4), (512, 4)],
        requires_grad=[True, False, False, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/276b7c300f72ab824c9a1087a0f7a11154e00564/mmdet/models/losses/smooth_l1_loss.py#L37",  # noqa
        tags=[SampleTag.Reduce, \
              SampleTag.IfElseBranch]
    )


def gen_np_args(M, N):
    def gen_base(num):
        data = np.random.randn(M, N) * 100
        data = data.astype(np.float32)
        return data

    pred = gen_base(N)
    target = gen_base(N)
    weight = gen_base(N)
    return [pred, target, weight, "sum"]


register_sample(__name__, get_sample_config, gen_np_args)
