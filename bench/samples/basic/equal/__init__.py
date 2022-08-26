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
    with open("./bench/samples/basic/equal/equal.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["equal_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["equal_0"][i], arg_data["equal_1"][i]))
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


def gen_np_args(equal_0, equal_1):
    equal_0 = np.random.random(equal_0)
    if isinstance(equal_1, list): # equal_1 is a tensor
        equal_1 = np.random.random(equal_1)
    else: # equal_1 is an imm
        equal_1 = np.array(equal_1)

    return [equal_0, equal_1]


register_sample(__name__, get_sample_config, gen_np_args)
