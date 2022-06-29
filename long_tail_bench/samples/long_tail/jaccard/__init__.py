from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(300, ), (600, ), (800, )],
        requires_grad=[False, False],
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.SINGLE_REPO,
        url="https://github.com/dbolya/yolact/blob/57b8f2d95e62e2e649b382f516ab41f949b57239/scripts/optimize_bboxes.py#L45",  # noqa
        tags=[SampleTag.Reduce, SampleTag.ViewAttribute],
    )


def gen_np_args(M):
    def gen_base(num):
        data = np.random.randn(num, 4) * 100
        data = data.astype(np.float32)
        return data

    box_a = gen_base(M)
    box_b = gen_base(M)

    return [box_a, box_b]


register_sample(__name__, get_sample_config, gen_np_args)
