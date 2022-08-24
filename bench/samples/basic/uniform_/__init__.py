from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/uniform_/uniform_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_tensor"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append([arg_data["input_tensor"][i], arg_data["low_bound"][i], arg_data["high_bound"][i]])
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_size_, low_bound_, high_bound_):
    input_np = np.random.random(input_size_)
    low_bound = low_bound_[0]
    high_bound = high_bound_[0]

    return [input_np, low_bound, high_bound]


register_sample(__name__, get_sample_config, gen_np_args)
