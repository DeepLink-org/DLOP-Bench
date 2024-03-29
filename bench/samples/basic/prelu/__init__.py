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
    with open("./bench/samples/basic/prelu/prelu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["prelu_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["prelu_0"][i], arg_data["prelu_1"][i]))
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


def gen_np_args(prelu_0, prelu_1):
    prelu_0 = np.random.random(prelu_0)
    if isinstance(prelu_1, list): # prelu_1 is a tensor
        prelu_1 = np.random.random(prelu_1)
    else: # prelu_1 is an imm
        prelu_1 = np.array(prelu_1)

    return [prelu_0, prelu_1]


register_sample(__name__, get_sample_config, gen_np_args)
