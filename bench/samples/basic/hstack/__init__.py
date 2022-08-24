from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import torch
import torch.nn

def get_sample_config():
    with open("./bench/samples/basic/hstack/hstack.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["out"][i]))
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

def gen_np_args(input_,out_):
    input_image_np = np.random.random(input_)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    out_image_np =np.ndarray(out_)
    out_image = torch.from_numpy(out_image_np).cuda()

    return [input_image, out_image]

register_sample(__name__, get_sample_config, gen_np_args)
