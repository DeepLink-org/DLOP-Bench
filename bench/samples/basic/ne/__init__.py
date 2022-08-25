from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/ne/ne.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["ne_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["ne_0"][i], arg_data["ne_1"][i]))
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


def gen_np_args(ne_0, ne_1):
    ne_0 = np.random.random(ne_0)
    if isinstance(ne_1, list): # ne_1 is a tensor
        ne_1 = np.random.random(ne_1)
    else: # ne_1 is an imm
        ne_1 = np.array(ne_1)

    return [ne_0, ne_1]


register_sample(__name__, get_sample_config, gen_np_args)
