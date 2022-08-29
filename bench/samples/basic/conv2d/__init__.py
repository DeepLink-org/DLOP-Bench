from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import random


def get_sample_config():
    with open("./long_tail_bench/samples/basic/conv2d/conv2d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    args_cases_ = []
    for i in range(arg_data_length):
        if random.random() < 0.01:
            args_cases_.append((arg_data["input_size"][i], arg_data["kernel_size"][i], arg_data["bias"][i], arg_data["stride"][i], arg_data["padding"][i], arg_data["dilation"][i], arg_data["groups"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 9,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_size_, kernel_size_, bias_, stride_, padding_, dilation_, groups_):
    in_channels = input_size_[1]
    out_channels = kernel_size_[0]
    input_image_np = np.random.random(input_size_)
    kernel_size = [kernel_size_[2], kernel_size_[3]]
    bias = bias_[0]
    stride = stride_
    padding = padding_
    dilation = dilation_
    groups = groups_[0]

    return [in_channels, out_channels, kernel_size, bias, stride, padding, dilation, groups, input_image_np]


register_sample(__name__, get_sample_config, gen_np_args)
