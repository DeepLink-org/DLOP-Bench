# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json

def get_sample_config():
    with open("./bench/samples/basic/conv3d/conv3d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["weight"][i], arg_data["bias"][i], arg_data["stride"][i], arg_data["padding"][i], arg_data["dilation"][i], arg_data["groups"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 7,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input, weight, bias, stride, padding, dilation, groups):
    input_image_np = np.random.random(input)
    weight_image_np = np.random.random(weight)
    if not bias[0]:
        bias_image_np = None
    else:
        bias_image_np = np.random.random(bias)
    
    stride_image = tuple(stride) if len(stride) > 1 else stride[0]
    padding_image = tuple(padding) if len(padding) > 1 else padding[0]
    dilation_image = tuple(dilation) if len(dilation) > 1 else dilation[0]
    groups_image = groups[0]
    return [input_image_np, weight_image_np, bias_image_np, stride_image, padding_image, dilation_image, groups_image]

register_sample(__name__, get_sample_config, gen_np_args)
