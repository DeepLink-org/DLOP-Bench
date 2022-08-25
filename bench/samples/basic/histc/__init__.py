from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/histc/histc.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input_size"][i], arg_data["bins"][i], arg_data["min"][i], arg_data["max"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 4,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_size_, bins_, min_, max_):
    input_np = np.random.random(input_size_)
    return [input_np, bins_, min_, max_]


register_sample(__name__, get_sample_config, gen_np_args)
