import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def conv2dt(input, weight, bias, stride, padding, output_padding, groups, dilation):
    input_image_np = np.random.random(input)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    weight_image_np = np.random.random(weight)
    weight_image = torch.from_numpy(weight_image_np).to(torch.float32).cuda()
    if not bias[0]:
        bias_image = None
    else:
        bias_image_np = np.random.random(bias)
        bias_image = torch.from_numpy(bias_image_np).to(torch.float32).cuda()
    
    stride_image = tuple(stride) if len(stride) > 1 else stride[0]
    padding_image = tuple(padding) if len(padding) > 1 else padding[0]
    output_padding_image = tuple(output_padding) if len(output_padding) > 1 else output_padding[0]
    groups_image = groups[0]
    dilation_image = tuple(dilation) if len(dilation) > 1 else dilation[0]
    ret = torch.nn.functional.conv_transpose2d(input_image, weight_image, bias_image, stride_image, padding_image, output_padding_image, groups_image, dilation_image).cuda()
    return ret


def args_adaptor(np_args):
    input = np_args[0]
    weight = np_args[1]
    bias = np_args[2]
    stride = np_args[3]
    padding = np_args[4]
    output_padding = np_args[5]
    groups = np_args[6]
    dilation = np_args[7]
    return [input, weight, bias, stride, padding, output_padding, groups, dilation]


def executer_creator():
    return Executer(conv2dt, args_adaptor)
