from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json

def get_sample_config():
    with open("./long_tail_bench/samples/basic/conv2dt/conv2dt.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["weight"][i], arg_data["bias"][i], arg_data["stride"][i], arg_data["padding"][i], arg_data["output_padding"][i], arg_data["groups"][i], arg_data["dilation"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 8,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input, weight, bias, stride, padding, output_padding, groups, dilation):
    return [input, weight, bias, stride, padding, output_padding, groups, dilation]


register_sample(__name__, get_sample_config, gen_np_args)
