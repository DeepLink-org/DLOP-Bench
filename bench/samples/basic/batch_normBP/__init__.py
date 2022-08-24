from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import random


def get_sample_config():
    with open("./bench/samples/basic/batch_norm/batchnorm.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["input"])
    args_cases_ = []
    for i in range(arg_data_length):
        if random.random() < 0.5:
            args_cases_.append((arg_data["input"][i], arg_data["running_mean"][i], 
                arg_data["running_var"][i], arg_data["weight"][i], arg_data["bias"][i], 
                arg_data["training"][i], arg_data["momentum"][i], arg_data["eps"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 8,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        url="",  # noqa
        tags=[],
    )


def gen_np_args(input_, running_mean_, running_var_,
            weight_, bias_, training_, momentum_, eps_):
    input = np.random.random(input_).astype(np.float32)
    running_mean = np.random.random(running_mean_).astype(np.float32)
    bias = np.random.random(bias_).astype(np.float32)
    running_var = np.random.random(running_var_).astype(np.float32)
    weight = np.random.random(weight_).astype(np.float32)
    training = training_
    momentum = momentum_
    eps = eps_

    return [input, running_mean, running_var, weight, bias,
            training, momentum, eps]


register_sample(__name__, get_sample_config, gen_np_args)
