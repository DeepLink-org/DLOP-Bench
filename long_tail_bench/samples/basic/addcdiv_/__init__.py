from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/addcdiv_/addcdiv_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input_size"][i], arg_data["tensor1_size"][i], arg_data["tensor2_size"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_size_, tensor1_size_, tensor2_size_):
    input_size = input_size_
    tensor1_size = tensor1_size_
    tensor2_size = tensor2_size_

    return [input_size, tensor1_size, tensor2_size]


register_sample(__name__, get_sample_config, gen_np_args)
