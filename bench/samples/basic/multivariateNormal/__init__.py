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
    with open("./bench/samples/basic/MultivariateNormal/MultivariateNormal.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["loc_size"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["loc_size"][i], arg_data["covariance_matrix_size"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=False,
        performance_iters=100,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )

def gen_np_args(loc_size_, covariance_matrix_size_):
    np_loc = np.random.randn(*tuple(loc_size_))
    lower_triangle_mat = np.tril(np.random.random(covariance_matrix_size_))
    eps = np.finfo(np.float32).eps
    np_cov_mat = lower_triangle_mat @ lower_triangle_mat.T + eps * np.ones(covariance_matrix_size_)
    return [np_loc, np_cov_mat]

register_sample(__name__, get_sample_config, gen_np_args)
