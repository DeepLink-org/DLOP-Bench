from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/normalize/normalize.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["p"][i], 
            arg_data["dim"][i], arg_data["eps"][i], arg_data["out"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 5,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        url="",  # noqa
        tags=[],
    )


def gen_np_args(input_, p_, dim_, eps_, out_):
    input = np.random.random(input_).astype(np.float32)

    return [input, p_, dim_, eps_, out_]


register_sample(__name__, get_sample_config, gen_np_args)
