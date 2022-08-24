from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/unfold/unfold.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["kernel_size"][i], arg_data["dilation"][i], arg_data["padding"][i], arg_data["stride"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 5,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input, kernel_size, dilation, padding, stride):
    input_image_np = np.random.random(input)
    kernel_size_image = tuple(kernel_size) if len(kernel_size) > 1 else kernel_size[0]
    dilation_image = tuple(dilation) if len(dilation) > 1 else dilation[0]
    padding_image = tuple(padding) if len(padding) > 1 else padding[0]
    stride_image = tuple(stride) if len(stride) > 1 else stride[0]
    return [input_image_np, kernel_size_image, dilation_image, padding_image, stride_image]


register_sample(__name__, get_sample_config, gen_np_args)
