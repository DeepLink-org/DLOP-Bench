from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/cat/cat.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["tensor0"])
    args_cases_ = []
    for i in range(arg_data_length):
        cur_case = []
        args_cases_.append([])
        for j in range(len(arg_data.keys())):
            if len(arg_data["tensor"+str(j)][i]) != 0:
                cur_case.append(arg_data["tensor"+str(j)][i])
        args_cases_[i] = [cur_case]
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 1,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input_tensor_):
    input_tensor = input_tensor_

    return [input_tensor]


register_sample(__name__, get_sample_config, gen_np_args)
