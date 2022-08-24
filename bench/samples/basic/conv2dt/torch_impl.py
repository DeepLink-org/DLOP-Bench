import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def conv2dt(input, weight, bias, stride, padding, output_padding, groups, dilation):
    ret = torch.nn.functional.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    weight_image = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    if np_args[2] is None:
        bias_image = None
    else:
        bias_image = torch.from_numpy(np_args[2]).to(torch.float32).cuda()
    
    return [input_image, weight_image, bias_image, np_args[3], np_args[4], np_args[5], np_args[6], np_args[7]]


def executer_creator():
    return Executer(conv2dt, args_adaptor)
