from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("/data/liuhangda/op_test/LongTail-Bench/long_tail_bench/samples/add/add.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["add1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["add1"][i], arg_data["add2"][i]))
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


def gen_np_args(add1, add2):
    add1 = np.random.random(add1)
    add2 = np.random.random(add2)

    return [add1, add2]


register_sample(__name__, get_sample_config, gen_np_args)
