from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/scatter_/scatter_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["x1"][i], arg_data["x2"][i], 
            arg_data["x3"][i], arg_data["x4"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 4,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        url="",  # noqa
        tags=[],
    )


def gen_np_args(dim_, index_, src_, reduce_):
    input = np.random.random(input_).astype(np.float32)
    target = np.random.randint(5, size=target_)

    return [input, target, weight_,
            size_average_, ignore_index_, reduce_, reduction_, label_smoothing_]


register_sample(__name__, get_sample_config, gen_np_args)
