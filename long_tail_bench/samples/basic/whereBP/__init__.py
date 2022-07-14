from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/whereBP/where.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["condition"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["condition"][i], arg_data["tensor1"][i], arg_data["tensor2"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(condition_size_, tensor1_size_, tensor2_size_):
    condition_np = np.random.random(condition_size_) > 0.5
    tensor1_np = np.random.random(tensor1_size_)
    tensor2_np = np.random.random(tensor2_size_)

    return [condition_np, tensor1_np, tensor2_np]


register_sample(__name__, get_sample_config, gen_np_args)
