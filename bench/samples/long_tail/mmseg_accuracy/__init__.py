from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(2, 19, 51, 104), (4, 38, 52, 124), (2, 9, 12, 208)],
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMSEG,
        url="https://github.com/open-mmlab/mmsegmentation/blob/504965184c3e6bc9ec43af54237129ef21981a5f/mmseg/models/losses/accuracy.py#L5",  # noqa
        tags=[
            SampleTag.ForLoop, SampleTag.IfElseBranch, SampleTag.ViewAttribute
        ])


def gen_np_args(M, N, M1, N1):
    pred = np.random.randn(M, N, M1, N1).astype(np.float32)
    target = np.random.randint(0, 256, (M, M1, N1), dtype=np.int64)
    pred_jax = np.random.randn(M, N).astype(np.float32)
    target_jax = np.random.randint(0, 256, (M, ), dtype=np.int64)
    return [pred, target, pred_jax, target_jax]


register_sample(__name__, get_sample_config, gen_np_args)
