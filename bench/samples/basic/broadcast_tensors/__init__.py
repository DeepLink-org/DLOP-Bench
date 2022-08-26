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
    with open("./bench/samples/basic/broadcast_tensors/broadcast_tensors.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_size"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_ = []
        for j in range(len(arg_data["input_size"][i])):
            args_.append(arg_data["input_size"][i][j])
        args_cases_.append([tuple(args_)])
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 1,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_size_):
    data = []
    for input_size in input_size_:
        data.append(np.random.random(input_size))
    return data


register_sample(__name__, get_sample_config, gen_np_args)
