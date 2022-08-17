from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/meshgrid/meshgrid.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input0"])
    args_cases_ = []
    cnt = 0
    for i in range(arg_data_length):
        cur_args = []
        for j in range(2): 
            cur_args.append(arg_data["input" + str(j)][i])
        if arg_data['input2'][i] != []:
            pass
        else:
            args_cases_.append(cur_args)
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(*inputs):
    args = []
    for i in range(2):
        args.append(np.random.random(inputs[i]))
    return args

register_sample(__name__, get_sample_config, gen_np_args)
