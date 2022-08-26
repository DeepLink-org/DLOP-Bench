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
    with open("./bench/samples/basic/stackBP/stack.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["dim"])
    args_cases_ = []
    for i in range(arg_data_length):
        cur_case = []
        args_cases_.append([arg_data["dim"][i]])
        for j in range(len(arg_data.keys())-1):
            if len(arg_data["tensor"+str(j)][i]) != 0:
                cur_case.append(arg_data["tensor"+str(j)][i])
        args_cases_[i].append(cur_case)
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )

def find_first_diff(input_size_):
    for i in range(len(input_size_[0])):
        cur = input_size_[0][i]
        for j in range(len(input_size_)):
            if input_size_[j][i] != cur:
                return i
    return 0

def gen_np_args(dim, input_size_):
    input_size = input_size_
    input_np = []
    
    axis_cal = 0
    for i in range(len(input_size)):
        input_tmp = np.random.random(input_size[i])
        input_np.append(input_tmp)
    if len(input_size_) > 1 and len(input_size_[0]): # not specify dim
        axis_cal = find_first_diff(input_size_)
    
    axis = axis_cal if dim[0]==False else dim[0]

    return [input_np, axis]


register_sample(__name__, get_sample_config, gen_np_args)

