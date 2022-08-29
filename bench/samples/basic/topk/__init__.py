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
    with open("./bench/samples/basic/topk/topk.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["topk_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["topk_0"][i], arg_data["topk_1"][i]))
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


def gen_np_args(topk_0, topk_1):
    topk_0 = np.random.random(topk_0)
    if isinstance(topk_1, list): # topk_1 is a tensor
        topk_1 = np.random.random(topk_1)
    else: # topk_1 is an imm
        topk_1 = np.array(topk_1)

    return [topk_0, topk_1]


register_sample(__name__, get_sample_config, gen_np_args)
