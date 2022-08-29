from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/copy_/copy_.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["copy__0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["copy__0"][i], arg_data["copy__1"][i]))
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


def gen_np_args(copy__0, copy__1):
    copy__0 = np.random.random(copy__0)
    if isinstance(copy__1, list): # copy__1 is a tensor
        copy__1 = np.random.random(copy__1)
    else: # copy__1 is an imm
        copy__1 = np.array(copy__1)

    return [copy__0, copy__1]


register_sample(__name__, get_sample_config, gen_np_args)
