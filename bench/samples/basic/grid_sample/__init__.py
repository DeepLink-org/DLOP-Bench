# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import random


def get_sample_config():
    with open("./bench/samples/basic/grid_sample/grid_sample.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        if random.random() < 0.6:
            args_cases_.append((arg_data["input"][i], arg_data["grid"][i], arg_data["mode"][i], arg_data["padding_mode"][i], arg_data["align_coners"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 5,
        backward=False,
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_size, grid_size, mode, padding_mode, align_coners):
    input_np = np.random.random(input_size)
    grid_np = np.random.random(grid_size)
    return [input_np, grid_np, mode, padding_mode, align_coners]


register_sample(__name__, get_sample_config, gen_np_args)
