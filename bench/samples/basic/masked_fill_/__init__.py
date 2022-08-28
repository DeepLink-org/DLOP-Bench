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
    with open("./bench/samples/basic/masked_fill_/masked_fill_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["masked"][i], arg_data["value"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=False,
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )

def gen_np_args(input_, masked_, value_):
    input_size = input_
    masked_size = masked_ 
    value = value_
    return [input_size, masked_size, value]

register_sample(__name__, get_sample_config, gen_np_args)
