from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/minBP/min.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["min1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["min1"][i], arg_data["min2"][i]))
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


def gen_np_args(min1, min2):
    min1 = np.random.random(min1)
    if len(min2) == 0: # no min2
        min2 = np.array(min2)
    elif isinstance(min2[0], int): # min2 is a tensor
        min2 = np.random.random(min2)
    else: # min2 is an imm
        min2 = np.array(min2)

    return [min1, min2]


register_sample(__name__, get_sample_config, gen_np_args)
