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
    with open("./bench/samples/basic/gt/gt.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["gt_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["gt_0"][i], arg_data["gt_1"][i]))
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


def gen_np_args(gt_0, gt_1):
    gt_0 = np.random.random(gt_0)
    if isinstance(gt_1, list): # gt_1 is a tensor
        gt_1 = np.random.random(gt_1)
    else: # gt_1 is an imm
        gt_1 = np.array(gt_1)

    return [gt_0, gt_1]


register_sample(__name__, get_sample_config, gen_np_args)
