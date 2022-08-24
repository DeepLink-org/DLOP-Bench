from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json

def get_sample_config():
    with open("./bench/samples/basic/avg_pool2dBP/avg_pool2d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["kernel_size"][i], \
            arg_data["stride"][i], arg_data["padding"][i], arg_data["ceil_mode"][i], \
                arg_data["count_include_pad"][i], arg_data["divisor_override"][i]))
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

def gen_np_args(input_, kernel_size_, stride_, padding_, ceil_mode_,\
    count_include_pad_, divisor_override_):
    input_image_np = np.random.random(input_)
    kernel_size = kernel_size_
    stride = stride_
    padding = padding_
    ceil_mode = ceil_mode_
    count_include_pad = count_include_pad_
    divisor_override = divisor_override_

    return [input_image_np, kernel_size, stride, padding, ceil_mode,\
        count_include_pad, divisor_override]

register_sample(__name__, get_sample_config, gen_np_args)
