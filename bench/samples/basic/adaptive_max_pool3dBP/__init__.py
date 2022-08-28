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
    with open("./bench/samples/basic/adaptive_max_pool3dBP/adaptive_max_pool3d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["output_size"][i],\
        arg_data["return_indices"][i]))
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

def gen_np_args(input_size, output_size_, return_indices_):
    input_image_np = np.random.random(input_size)
    output_size = output_size_
    return_indices = return_indices_
    return [input_image_np, output_size, return_indices]

register_sample(__name__, get_sample_config, gen_np_args)
