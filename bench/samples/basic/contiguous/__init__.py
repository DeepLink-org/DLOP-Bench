from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import random


def get_sample_config():
    with open("./long_tail_bench/samples/basic/contiguous/contiguous.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_shape"])
    args_cases_ = []
    for i in range(arg_data_length):
        if random.random() < 0.3:
            args_cases_.append([arg_data["input_shape"][i]])
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 1,
        backward=[False],
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_shape):
    input_tensor = np.random.random(input_shape)
    return [input_tensor]


register_sample(__name__, get_sample_config, gen_np_args)
