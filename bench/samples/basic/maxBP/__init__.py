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
    with open("./bench/samples/basic/maxBP/max.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["max1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["max1"][i], arg_data["max2"][i]))
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


def gen_np_args(max1, max2):
    max1 = np.random.random(max1)
    if len(max2) == 0: # no max2
        max2 = np.array(max2)
    elif isinstance(max2[0], int): # max2 is a tensor
        max2 = np.random.random(max2)
    else: # max2 is an imm
        max2 = np.array(max2)

    return [max1, max2]


register_sample(__name__, get_sample_config, gen_np_args)
