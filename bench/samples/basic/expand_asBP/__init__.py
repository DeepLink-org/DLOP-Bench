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
    with open("./bench/samples/basic/expand_as/expand_as.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_shape"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input_shape"][i], arg_data["expand_shape"][i]))
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


def gen_np_args(add1, add2):
    add1 = np.random.random(add1)
    if isinstance(add2[0], int): # add2 is a tensor
        add2 = np.random.random(add2)
    else: # add2 is an imm
        add2 = np.array(add2)

    return [add1, add2]


register_sample(__name__, get_sample_config, gen_np_args)
