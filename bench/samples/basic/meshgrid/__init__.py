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
    with open("./bench/samples/basic/meshgrid/meshgrid.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input0"])
    args_cases_ = []
    for i in range(arg_data_length):
        cur_args = []
        for j in range(len(arg_data)): 
            cur_args.append(arg_data["input" + str(j)][i])
        args_cases_.append(cur_args)
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * len(arg_data),
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(*inputs):
    args = []
    for input_ in inputs:
        args.append(np.random.random(input_))
    return args

register_sample(__name__, get_sample_config, gen_np_args)
