from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import random


def get_sample_config():
    with open("./long_tail_bench/samples/basic/einsum/einsum.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["x1"][i], arg_data["x2"][i], 
            arg_data["x3"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        url="",  # noqa
        tags=[],
    )


def gen_np_args(equation_, op1_, op2_):
    equation = equation_.replace('/', ',')
    op1 = np.random.random(op1_).astype(np.float32)
    op2 = np.random.random(op2_).astype(np.float32)
    return [equation, op1, op2]


register_sample(__name__, get_sample_config, gen_np_args)
