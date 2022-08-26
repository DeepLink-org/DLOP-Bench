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
    with open("./bench/samples/basic/margin_ranking_loss/margin_ranking_loss.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input1"][i], arg_data["input2"][i], arg_data["target"][i], arg_data["margin"][i], arg_data["reduction"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 5,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input1_, input2_, target_, margin_, redu_):
    input1_np = np.random.random(input1_)
    input2_np = np.random.random(input2_)
    target_np = np.random.random(target_)
    return [input1_np, input2_np, target_np, margin_, redu_]


register_sample(__name__, get_sample_config, gen_np_args)
