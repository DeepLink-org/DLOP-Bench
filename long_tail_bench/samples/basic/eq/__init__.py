from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/eq/eq.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["eq_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["eq_0"][i], arg_data["eq_1"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(eq_0, eq_1):
    eq_0 = np.random.random(eq_0)
    if isinstance(eq_1, list): # eq_1 is a tensor
        eq_1 = np.random.random(eq_1)
    else: # eq_1 is an imm
        eq_1 = np.array(eq_1)

    return [eq_0, eq_1]


register_sample(__name__, get_sample_config, gen_np_args)
