import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def linear(input, weight, bias):
    ret = torch.nn.functional.linear(input, weight, bias)
    ret.backward(ret)
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input_image.requires_grad = True

    weight_image = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    weight_image.requires_grad = True

    if np_args[2] is None:
        bias_image = None
    else:
        bias_image = torch.from_numpy(np_args[2]).to(torch.float32).cuda()
        bias_image.requires_grad = True

    return [input_image, weight_image, bias_image]


def executer_creator():
    return Executer(linear, args_adaptor)
