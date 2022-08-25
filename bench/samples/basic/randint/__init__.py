from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json

def get_sample_config():
    with open("./bench/samples/basic/randint/randint.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["low"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["low"][i], arg_data["high"][i], arg_data["size"][i]))
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

def gen_np_args(low_, high_, size_):
    low = low_
    high = high_
    size = tuple(size_)
    return [low, high, size]

register_sample(__name__, get_sample_config, gen_np_args)
