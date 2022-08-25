from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import random


def get_sample_config():
    with open("./bench/samples/basic/layer_norm/layer_norm.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["x1"][i], arg_data["x2"][i], 
            arg_data["x3"][i], arg_data["x4"][i], arg_data["x5"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 5,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        url="",  # noqa
        tags=[],
    )


def gen_np_args(input_, normalized_shape_, weight_, bias_, eps_):
    input = np.random.random(input_).astype(np.float32)
    weight = np.random.random(weight_).astype(np.float32)
    bias = np.random.random(bias_).astype(np.float32)
    return [input, normalized_shape_, weight, bias, eps_]


register_sample(__name__, get_sample_config, gen_np_args)
