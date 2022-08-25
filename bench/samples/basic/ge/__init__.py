from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/ge/ge.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["ge_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["ge_0"][i], arg_data["ge_1"][i]))
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


def gen_np_args(ge_0, ge_1):
    ge_0 = np.random.random(ge_0)
    if isinstance(ge_1, list): # ge_1 is a tensor
        ge_1 = np.random.random(ge_1)
    else: # ge_1 is an imm
        ge_1 = np.array(ge_1)

    return [ge_0, ge_1]


register_sample(__name__, get_sample_config, gen_np_args)
