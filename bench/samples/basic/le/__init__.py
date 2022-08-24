from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/le/le.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["le_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["le_0"][i], arg_data["le_1"][i]))
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


def gen_np_args(le_0, le_1):
    le_0 = np.random.random(le_0)
    if isinstance(le_1, list): # le_1 is a tensor
        le_1 = np.random.random(le_1)
    else: # le_1 is an imm
        le_1 = np.array(le_1)

    return [le_0, le_1]


register_sample(__name__, get_sample_config, gen_np_args)
