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
    with open("./bench/samples/basic/interpolate/interpolate.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    args_cases_ = []
    for i in range(arg_data_length):
        if arg_data["output_size"][i][0] != False or arg_data["scale_factor"][i][0] != False or arg_data["scale_factor"][i][0] != False:
            args_cases_.append((arg_data["input_size"][i], arg_data["output_size"][i], arg_data["scale_factor"][i], arg_data["mode"][i], arg_data["align_corners"][i], arg_data["recompute_scale_factor"][i], arg_data["antialias"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 6,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_size_, output_size_, scale_factor_, mode_, align_corners_, recompute_scale_factor_, antialias_):
    input_np = np.random.random([int(i) for i in input_size_])
    if output_size_[0] != False:
        output_size = [int(i) for i in output_size_]
    else:
        output_size = None

    if scale_factor_[0] == False:
        scale_factor = None
    else:
        if len(scale_factor_) > 1:
            scale_factor = [float(i) for i in scale_factor_]
        else:
            scale_factor = float(scale_factor_[0])

    mode = mode_[0]
    if mode_[0] == "None":
        mode = "nearest"

    if mode in ["linear", "bilinear", "bicubic",  "trilinear"]:
        if align_corners_[0] == 'False':
            align_corners = False
        else:
            align_corners = True
    else:
        align_corners = None

    if recompute_scale_factor_[0] == 'None':
        recompute_scale_factor = False
    else:
        recompute_scale_factor = recompute_scale_factor_[0]

    antialias = antialias_

    return [input_np, output_size, scale_factor, mode, align_corners, recompute_scale_factor]


register_sample(__name__, get_sample_config, gen_np_args)
