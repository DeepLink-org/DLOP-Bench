from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/log_softmax/log_softmax.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["log_softmax_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["log_softmax_0"][i], arg_data["log_softmax_1"][i]))
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


def gen_np_args(log_softmax_0, log_softmax_1):
    log_softmax_0 = np.random.random(log_softmax_0)
    if isinstance(log_softmax_1, list): # log_softmax_1 is a tensor
        log_softmax_1 = np.random.random(log_softmax_1)
    else: # log_softmax_1 is an imm
        log_softmax_1 = np.array(log_softmax_1)

    return [log_softmax_0, log_softmax_1]


register_sample(__name__, get_sample_config, gen_np_args)
