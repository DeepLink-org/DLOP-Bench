import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def unfold(input, kernel_size, dilation, padding, stride):
    input_image_np = np.random.random(input)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    kernel_size_image = tuple(kernel_size) if len(kernel_size) > 1 else kernel_size[0]
    dilation_image = tuple(dilation) if len(dilation) > 1 else dilation[0]
    padding_image = tuple(padding) if len(padding) > 1 else padding[0]
    stride_image = tuple(stride) if len(stride) > 1 else stride[0]
    ret = torch.nn.functional.unfold(input_image, kernel_size_image, dilation_image, padding_image, stride_image)
    return ret


def args_adaptor(np_args):
    input = np_args[0]
    kernel_size = np_args[1]
    dilation = np_args[2]
    padding = np_args[3]
    stride = np_args[4]
    return [input, kernel_size, dilation, padding, stride]


def executer_creator():
    return Executer(unfold, args_adaptor)
