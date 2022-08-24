from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/baddbmmBP/baddbmm.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["batch1"][i], arg_data["batch2"][i]))
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


def gen_np_args(input, batch1, batch2):
    input_image_np = np.random.random(input)
    batch1_image_np = np.random.random(batch1)
    batch2_image_np = np.random.random(batch2)
    return [input_image_np, batch1_image_np, batch2_image_np]


register_sample(__name__, get_sample_config, gen_np_args)
