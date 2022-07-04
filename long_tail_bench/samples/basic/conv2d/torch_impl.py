import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def conv2d(n, in_channels, out_channels, kernel_size, bias, stride, padding, dilation, groups):
    input_image_np = np.random.random([n, in_channels, kernel_size[0], kernel_size[1]])
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).cuda()
    ret = conv2d(input_image)
    return ret


def args_adaptor(np_args):
    n = np_args[0][0]
    in_channels = np_args[0][1]
    out_channels = np_args[1][0]
    kernel_size = [np_args[1][2], np_args[1][3]]
    bias = np_args[2][0]
    stride = np_args[3]
    padding = np_args[4]
    dilation = np_args[5]
    groups = np_args[6][0]
    return [n, in_channels, out_channels, kernel_size, bias, stride, padding, dilation, groups]


def executer_creator():
    return Executer(conv2d, args_adaptor)
