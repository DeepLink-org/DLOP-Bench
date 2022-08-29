# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json

def get_sample_config():
    with open("./bench/samples/basic/embedding/embedding.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["input"][i], arg_data["weight"][i], arg_data["padding_idx"][i], arg_data["max_norm"][i], arg_data["norm_type"][i], arg_data["scale_grad_by_freq"][i], arg_data["sparse"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 7,
        backward=False,
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    input_image_np = np.random.random(input)
    weight_image_np = np.random.random(weight)
    padding_idx_image = padding_idx[0]
    max_norm_image = max_norm[0]
    norm_type_image = norm_type[0]
    scale_grad_by_freq_image = scale_grad_by_freq[0]
    sparse_iamge = sparse[0]
    return [input_image_np, weight_image_np, padding_idx_image, max_norm_image, norm_type_image, scale_grad_by_freq_image, sparse_iamge]


register_sample(__name__, get_sample_config, gen_np_args)
