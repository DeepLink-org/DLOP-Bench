from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/arange/arange.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["start"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["start"][i], arg_data["end"][i], arg_data["step"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(start_, end_, step_):
    start = start_[0]
    end = end_[0]
    step = step_[0]
    
    return [start, end, step]

register_sample(__name__, get_sample_config, gen_np_args)
