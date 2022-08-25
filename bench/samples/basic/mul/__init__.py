from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/mul/mul.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["mul_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["mul_0"][i], arg_data["mul_1"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(mul_0, mul_1):
    mul_0 = np.random.random(mul_0)
    if isinstance(mul_1, list): # mul_1 is a tensor
        mul_1 = np.random.random(mul_1)
    else: # mul_1 is an imm
        mul_1 = np.array(mul_1)

    return [mul_0, mul_1]


register_sample(__name__, get_sample_config, gen_np_args)
