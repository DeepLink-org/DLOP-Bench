from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(12,), (16,), (32,)],
        requires_grad=[False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.OPENNMT,
        url="https://github.com/OpenNMT/OpenNMT-py/blob/c8081afa3fab4e169c34a84f9304cf37184f97d7/onmt/utils/misc.py#L56",  # noqa
        tags=[
            SampleTag.InputAware, SampleTag.Broadcast, SampleTag.ViewAttribute
        ],
    )


def gen_np_args(N,):
    shape = (N, )
    input1 = np.random.randint(0, 5, shape)
    input1 = input1.astype(np.float32)
    return [input1]


register_sample(__name__, get_sample_config, gen_np_args)
