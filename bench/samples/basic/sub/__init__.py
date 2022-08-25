from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/sub/sub.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["sub_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["sub_0"][i], arg_data["sub_1"][i]))
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


def gen_np_args(sub_0, sub_1):
    sub_0 = np.random.random(sub_0)
    if isinstance(sub_1, list): # sub_1 is a tensor
        sub_1 = np.random.random(sub_1)
    else: # sub_1 is an imm
        sub_1 = np.array(sub_1)

    return [sub_0, sub_1]


register_sample(__name__, get_sample_config, gen_np_args)
