from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(12, ), (16, ), (32, )],
        requires_grad=[False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.OPENNMT,
        url="https://github.com/PaddlePaddle/PaddleRec/blob/8ce9dbdbc5d0149bfcdb57a06a183024cff21aa3/models/recall/mind/net.py#L152",  # noqa
        tags=[
            SampleTag.Reduce
        ],
    )


def gen_np_args(N):
    shape = (N, )
    input1 = np.random.randint(0, 5, shape)
    input1 = input1.astype(np.float32)
    return [input1]


register_sample(__name__, get_sample_config, gen_np_args)
