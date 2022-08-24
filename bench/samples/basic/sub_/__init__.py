from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/sub_/sub_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["sub__0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["sub__0"][i], arg_data["sub__1"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(sub__0, sub__1):
    sub__0 = np.random.random(sub__0)
    if isinstance(sub__1, list): # sub__1 is a tensor
        sub__1 = np.random.random(sub__1)
    else: # sub__1 is an imm
        sub__1 = np.array(sub__1)

    return [sub__0, sub__1]


register_sample(__name__, get_sample_config, gen_np_args)
