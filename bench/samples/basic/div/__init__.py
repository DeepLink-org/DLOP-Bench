from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/div/div.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["div_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["div_0"][i], arg_data["div_1"][i]))
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


def gen_np_args(div_0, div_1):
    div_0 = np.random.random(div_0)
    if isinstance(div_1, list): # div_1 is a tensor
        div_1 = np.random.random(div_1)
    else: # div_1 is an imm
        div_1 = np.array(div_1)

    return [div_0, div_1]


register_sample(__name__, get_sample_config, gen_np_args)
