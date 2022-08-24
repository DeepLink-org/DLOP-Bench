from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import torch

def get_sample_config():
    with open("./bench/samples/basic/adaptive_avg_pool3d/adaptive_avg_pool3d.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["output_size"][i]))
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

def gen_np_args(input_, output_size_):
    input_image_np = np.random.random(input_)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    output_size = output_size_
    return [input_image, output_size]

register_sample(__name__, get_sample_config, gen_np_args)