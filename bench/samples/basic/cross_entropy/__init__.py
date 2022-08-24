from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/cross_entropy/cross_entropy.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["target"][i], 
            arg_data["weight"][i], arg_data["size_average"][i], arg_data["ignore_index"][i], 
            arg_data["reduce"][i], arg_data["reduction"][i], arg_data["label_smoothing"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 8,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        url="",  # noqa
        tags=[],
    )


def gen_np_args(input_, target_, weight_,
            size_average_, ignore_index_, reduce_, reduction_, label_smoothing_):
    input = np.random.random(input_).astype(np.float32)
    target = np.random.randint(5, size=target_)

    return [input, target, weight_,
            size_average_, ignore_index_, reduce_, reduction_, label_smoothing_]


register_sample(__name__, get_sample_config, gen_np_args)
