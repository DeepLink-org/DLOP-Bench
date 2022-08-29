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
    with open("./bench/samples/basic/affine_grid/affine_grid.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["theta"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["theta"][i], arg_data["size"][i], arg_data["align_corners"][i]))
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


def gen_np_args(theta, size, align_corners):
    theta_np = np.random.random(theta)
    return [theta_np, size, align_corners]


register_sample(__name__, get_sample_config, gen_np_args)
