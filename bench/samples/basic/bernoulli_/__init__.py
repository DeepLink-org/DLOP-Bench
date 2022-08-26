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
    with open("./bench/samples/basic/bernoulli_/bernoulli_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["size"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["size"][i], arg_data["p"][i]))
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

def gen_np_args(size_, p_):
    np_input = np.random.randint(0, 100, size_)
    if isinstance(p_, list):
        np_p = np.random.uniform(0.0, 1.0, tuple(p_))
    else:
        np_p = p_
    return [np_input, np_p]

register_sample(__name__, get_sample_config, gen_np_args)
