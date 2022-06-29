from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np

# center_size results:
# parrots: s1 ~ s5 pass


def get_sample_config():
    return SampleConfig(
        args_cases=[(10, ), (20, ), (30, )],
        requires_grad=[False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/dbolya/yolact/blob/57b8f2d95e62e2e649b382f516ab41f949b57239/layers/box_utils.py#L21",  # noqa
        tags=[
            SampleTag.ViewAttribute
        ],
    )


def gen_np_args(N):
    def gen_base(num):
        data = np.random.randn(num, 4) * 100
        data = data.astype(np.int)
        return data

    boxes = gen_base(N)
    return [boxes]


register_sample(__name__, get_sample_config, gen_np_args)
