from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/logical_xor/logical_xor.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["logical_xor_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["logical_xor_0"][i], arg_data["logical_xor_1"][i]))
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


def gen_np_args(logical_xor_0, logical_xor_1):
    logical_xor_0 = np.random.random(logical_xor_0)
    if isinstance(logical_xor_1, list): # logical_xor_1 is a tensor
        logical_xor_1 = np.random.random(logical_xor_1)
    else: # logical_xor_1 is an imm
        logical_xor_1 = np.array(logical_xor_1)

    return [logical_xor_0, logical_xor_1]


register_sample(__name__, get_sample_config, gen_np_args)
