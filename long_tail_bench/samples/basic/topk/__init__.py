from long_tail_bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./long_tail_bench/samples/basic/topk/topk.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["topk_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["topk_0"][i], arg_data["topk_1"][i], arg_data["topk_2"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 3,
        backward=[False] * 2,
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(topk_0, topk_1, topk_2):
    topk_0 = np.random.random(topk_0)
    topk_1 = np.array(topk_1)
    topk_2 = np.array(topk_2)

    return [topk_0, topk_1, topk_2]


register_sample(__name__, get_sample_config, gen_np_args)
