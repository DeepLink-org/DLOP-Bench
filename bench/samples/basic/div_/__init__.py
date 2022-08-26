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
    with open("./bench/samples/basic/div_/div_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["div__0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["div__0"][i], arg_data["div__1"][i]))
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


def gen_np_args(div__0, div__1):
    div__0 = np.random.random(div__0)
    if isinstance(div__1, list): # div__1 is a tensor
        div__1 = np.random.random(div__1)
    else: # div__1 is an imm
        div__1 = np.array(div__1)

    return [div__0, div__1]


register_sample(__name__, get_sample_config, gen_np_args)
