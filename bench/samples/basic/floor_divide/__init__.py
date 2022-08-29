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
    with open("./bench/samples/basic/floor_divide/floor_divide.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["floor_divide_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["floor_divide_0"][i], arg_data["floor_divide_1"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=False,
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(floor_divide_0, floor_divide_1):
    floor_divide_0 = np.random.random(floor_divide_0)
    if isinstance(floor_divide_1, list): # floor_divide_1 is a tensor
        floor_divide_1 = np.random.random(floor_divide_1)
    else: # floor_divide_1 is an imm
        floor_divide_1 = np.array(floor_divide_1)

    return [floor_divide_0, floor_divide_1]


register_sample(__name__, get_sample_config, gen_np_args)
