# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def conv2d(op, input_torch):
    ret = op(input_torch)
    return ret


def args_adaptor(np_args):
    in_channels = np_args[0]
    out_channels = np_args[1]
    kernel_size = np_args[2]
    bias = np_args[3]
    stride = np_args[4]
    padding = np_args[5]
    dilation = np_args[6]
    groups = np_args[7]
    input_torch = torch.from_numpy(np_args[8]).to(torch.float32).cuda()
    conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).cuda()
    
    return [conv2d, input_torch]


def executer_creator():
    return Executer(conv2d, args_adaptor)
