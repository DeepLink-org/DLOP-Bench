from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/sparse_coo_tensor/sparse_coo_tensor.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["indices"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["indices"][i], arg_data["values"][i], arg_data["size"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=False,
        performance_iters=100, # bug : crush when equal 1000
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(indices, values, size):
    indices_tensor = np.random.random(indices)
    values_tensor = np.random.random(values)

    return [indices_tensor, values_tensor, size]


register_sample(__name__, get_sample_config, gen_np_args)
