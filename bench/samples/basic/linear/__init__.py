from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/linear/linear.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["weight"][i], arg_data["bias"][i]))
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


def gen_np_args(input_, weight_, bias_):
    input_image_np = np.random.random(input_)
    weight_image_np = np.random.random(weight_)
    if not bias_[0]:
        bias_image_np = None
    else:
        bias_image_np = np.random.random(bias_)
    return [input_image_np, weight_image_np, bias_image_np]


register_sample(__name__, get_sample_config, gen_np_args)
