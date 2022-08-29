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
    with open("./bench/samples/basic/pow/pow.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["pow_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["pow_0"][i], arg_data["pow_1"][i]))
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


def gen_np_args(pow_0, pow_1):
    pow_0 = np.random.random(pow_0)
    if isinstance(pow_1, list): # pow_1 is a tensor
        pow_1 = np.random.random(pow_1)
    else: # pow_1 is an imm
        pow_1 = np.array(pow_1)

    return [pow_0, pow_1]


register_sample(__name__, get_sample_config, gen_np_args)
