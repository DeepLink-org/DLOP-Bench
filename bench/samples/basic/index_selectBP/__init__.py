# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import random


def get_sample_config():
    with open("./bench/samples/basic/index_selectBP/index_select.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        if random.random() < 0.5:
            args_cases_.append((arg_data["input"][i], arg_data["dim"][i], arg_data["index"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=False,
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )

def gen_np_args(input_size_, dim_, index_):
    input_image_np = np.random.random(input_size_)
    index_tensor = np.random.randint(0, input_size_[int(dim_)], (int(index_[0]),))
    return [input_image_np, dim_, index_tensor]

register_sample(__name__, get_sample_config, gen_np_args)
