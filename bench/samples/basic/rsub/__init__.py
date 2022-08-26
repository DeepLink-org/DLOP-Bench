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
    with open("./bench/samples/basic/rsub/rsub.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["rsub_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["rsub_0"][i], arg_data["rsub_1"][i]))
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


def gen_np_args(rsub_0, rsub_1):
    rsub_0 = np.random.random(rsub_0)
    if isinstance(rsub_1, list): # rsub_1 is a tensor
        rsub_1 = np.random.random(rsub_1)
    else: # rsub_1 is an imm
        rsub_1 = np.array(rsub_1)

    return [rsub_0, rsub_1]


register_sample(__name__, get_sample_config, gen_np_args)
