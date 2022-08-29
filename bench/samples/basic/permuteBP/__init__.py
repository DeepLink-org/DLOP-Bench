from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/permuteBP/permute.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input_tensor"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input_tensor"][i], arg_data["dims"][i]))
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


def gen_np_args(input_size_, dims_):
    input_np = np.random.random(input_size_)
    dims = dims_

    return [input_np, dims]


register_sample(__name__, get_sample_config, gen_np_args)
