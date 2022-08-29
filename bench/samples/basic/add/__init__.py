from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/add/add.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["add1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["add1"][i], arg_data["add2"][i]))
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


def gen_np_args(add1_size, add2_size):
    add1_np = np.random.random(add1_size)
    if isinstance(add2_size[0], int): # add2 is a tensor
        add2_np = np.random.random(add2_size)
    else: # add2 is an imm
        add2_np = np.array(add2_size)

    return [add1_np, add2_np]


register_sample(__name__, get_sample_config, gen_np_args)
