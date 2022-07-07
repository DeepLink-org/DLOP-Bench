import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def linear(input, weight, bias):
    input_image_np = np.random.random(input)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    weight_image_np = np.random.random(weight)
    weight_image = torch.from_numpy(weight_image_np).to(torch.float32).cuda()
    if not bias[0]:
        bias_image = None
    else:
        bias_image_np = np.random.random(bias)
        bias_image = torch.from_numpy(bias_image_np).to(torch.float32).cuda()
    ret = torch.nn.functional.linear(input_image, weight_image, bias_image).cuda()
    return ret


def args_adaptor(np_args):
    input = np_args[0]
    weight = np_args[1]
    bias = np_args[2]
    return [input, weight, bias]


def executer_creator():
    return Executer(linear, args_adaptor)
