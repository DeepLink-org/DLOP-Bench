from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import torch
import torch.nn

def get_sample_config():
    with open("./long_tail_bench/samples/basic/max_unpool2dBP/max_unpool2d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["indices"][i], \
            arg_data["kernel_size"][i], arg_data["stride"][i], \
            arg_data["padding"][i], arg_data["output_size"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 6,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )

def gen_np_args(input_, indices_, kernel_size_, stride_, padding_, output_size_):
    input_image = input_
    indices_image = indices_
    kernel_size = kernel_size_
    stride = stride_
    padding = padding_
    output_size = output_size_

    return [input_image, indices_image, kernel_size, stride, padding, output_size]

register_sample(__name__, get_sample_config, gen_np_args)
