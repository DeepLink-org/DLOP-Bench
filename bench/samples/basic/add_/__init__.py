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
    with open("./bench/samples/basic/add_/add_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["add__0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["add__0"][i], arg_data["add__1"][i]))
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


def gen_np_args(add__0_size, add__1_size):
    add__0_np = np.random.random(add__0_size)
    if isinstance(add__1_size, list): # add__1 is a tensor
        add__1_np = np.random.random(add__1_size)
    else: # add__1 is an imm
        add__1_np = np.array(add__1_size)

    return [add__0_np, add__1_np]


register_sample(__name__, get_sample_config, gen_np_args)
