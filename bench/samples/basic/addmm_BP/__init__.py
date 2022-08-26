# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import torch
import torch.nn

def get_sample_config():
    with open("./bench/samples/basic/addmm_BP/addmm_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["mat1"][i], \
            arg_data["mat2"][i]))
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

def gen_np_args(input_, mat1_, mat2_ ):
    input_image_np = np.random.random(input_)
    mat1_np =  np.random.random(mat1_)
    mat2_np = np.random.random(mat2_)

    return [input_image_np, mat1_np, mat2_np]

register_sample(__name__, get_sample_config, gen_np_args)
