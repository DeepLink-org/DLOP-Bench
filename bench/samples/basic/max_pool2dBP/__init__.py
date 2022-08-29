# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import torch

def get_sample_config():
    with open("./bench/samples/basic/max_pool2dBP/max_pool2d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["kernel_size"][i],\
            arg_data["stride"][i], arg_data["padding"][i],\
            arg_data["dilation"][i], arg_data["ceil_mode"][i],\
            arg_data["return_indices"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 7,
        backward=False,
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )

def gen_np_args(input_, kernel_size_, stride_, padding_, dilation_, ceil_mode_,\
    return_indices_):
    input_image_np = np.random.random(input_)
    kernel_size = kernel_size_
    stride = stride_
    padding = padding_
    dilation = dilation_
    ceil_mode = ceil_mode_
    return_indices = return_indices_

    return [input_image_np, kernel_size, stride, padding, dilation, ceil_mode,\
        return_indices]

register_sample(__name__, get_sample_config, gen_np_args)
