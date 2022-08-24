from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(16, 32), (16, 16), (32, 32)],
        requires_grad=[False, False],
        backward=[False],
        save_timeline=False,
        source=SampleSource.SEGMENTBASE2,
        url="https://gitlab.bj.sensetime.com/parrots.fit/segmentbase2/-/blob/master/segmentbase2/models/losses/L2_loss.py#L27",  # noqa
        tags=[SampleTag.Reduce]
    )


def gen_np_args(M, N):
    output = np.random.randn(M, N).astype(np.float32)
    target = np.random.randn(M, N).astype(np.float32)
    return [output, target]


register_sample(__name__, get_sample_config, gen_np_args)
