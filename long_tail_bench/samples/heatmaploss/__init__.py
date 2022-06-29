from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(1, 3), (2, 6), (4, 8)],
        requires_grad=[True, False, False],
        backward=[True],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMPOSE,
        url="https://github.com/open-mmlab/mmpose/blob/e8165f7c0c0590f84243120318ba199fd5c31e41/mmpose/models/losses/multi_loss_factory.py#L30",  # noqa
        tags=[SampleTag.ViewAttribute,
              SampleTag.Reduce]
    )


def gen_np_args(M, N):
    def gen_base(row, column):
        data = np.random.randn(row, column, 64, 64) * 100
        data = data.astype(np.float32)
        return data

    def gen_mask(row):
        data = np.zeros((row, 64, 64))
        data = data.astype(np.float32)
        return data

    pred = gen_base(M, N)
    gt = gen_base(M, N)
    mask = gen_mask(M)

    return [pred, gt, mask]


register_sample(__name__, get_sample_config, gen_np_args)
