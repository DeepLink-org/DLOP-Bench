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
    with open("./bench/samples/basic/is_neg/is_neg.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["is_neg_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["is_neg_0"][i], ))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 1,
        backward=False,
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(is_neg_0):
    is_neg_0 = np.random.random(is_neg_0)
    return [is_neg_0]


register_sample(__name__, get_sample_config, gen_np_args)