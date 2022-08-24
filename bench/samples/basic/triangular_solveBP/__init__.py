from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/triangular_solveBP/triangular_solve.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["b"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["b"][i], arg_data["A"][i]))
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


def gen_np_args(b, A):
    b_image_np = np.random.random(b)
    A_image_np = np.random.random(A)
    return [b_image_np, A_image_np]


register_sample(__name__, get_sample_config, gen_np_args)
