from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/relu/relu.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["relu_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["relu_0"][i], ))
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


def gen_np_args(relu_0):
    relu_0 = np.random.random(relu_0)
    return [relu_0]


register_sample(__name__, get_sample_config, gen_np_args)