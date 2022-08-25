from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/lstsq/lstsq.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_shape"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append([arg_data["input_shape"][i], arg_data["A_shape"][i]])
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=False,0, # bug : crush when equal 1000
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_shape, A_shape):
    input_tensor = np.random.random(input_shape)
    A_tensor = np.random.random(input_shape)

    return [input_tensor, A_tensor]


register_sample(__name__, get_sample_config, gen_np_args)
