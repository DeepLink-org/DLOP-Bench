from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/mul_/mul_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["mul__0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["mul__0"][i], arg_data["mul__1"][i]))
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


def gen_np_args(mul__0, mul__1):
    mul__0 = np.random.random(mul__0)
    if isinstance(mul__1, list): # mul__1 is a tensor
        mul__1 = np.random.random(mul__1)
    else: # mul__1 is an imm
        mul__1 = np.array(mul__1)

    return [mul__0, mul__1]


register_sample(__name__, get_sample_config, gen_np_args)
