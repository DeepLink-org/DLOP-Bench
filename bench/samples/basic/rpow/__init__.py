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
    with open("./bench/samples/basic/rpow/rpow.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["rpow_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["rpow_0"][i], arg_data["rpow_1"][i]))
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


def gen_np_args(rpow_0, rpow_1):
    rpow_0 = np.random.random(rpow_0)
    if isinstance(rpow_1, list): # rpow_1 is a tensor
        rpow_1 = np.random.random(rpow_1)
    else: # rpow_1 is an imm
        rpow_1 = np.array(rpow_1)

    return [rpow_0, rpow_1]


register_sample(__name__, get_sample_config, gen_np_args)
