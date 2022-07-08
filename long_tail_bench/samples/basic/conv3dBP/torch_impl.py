import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def conv3d(input, weight, bias, stride, padding, dilation, groups):
    input_image_np = np.random.random(input)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    input_image.requires_grad = True
    weight_image_np = np.random.random(weight)
    weight_image = torch.from_numpy(weight_image_np).to(torch.float32).cuda()
    weight_image.requires_grad = True
    if not bias[0]:
        bias_image = None
    else:
        bias_image_np = np.random.random(bias)
        bias_image = torch.from_numpy(bias_image_np).to(torch.float32).cuda()
        bias_image.requires_grad = True
    stride_image = tuple(stride) if len(stride) > 1 else stride[0]
    padding_image = tuple(padding) if len(padding) > 1 else padding[0]
    dilation_image = tuple(dilation) if len(dilation) > 1 else dilation[0]
    groups_image = groups[0]
    ret = torch.nn.functional.conv3d(input_image, weight_image, bias_image, stride_image, padding_image, dilation_image, groups_image)
    ret.backward(ret)
    return ret


def args_adaptor(np_args):
    input = np_args[0]
    weight = np_args[1]
    bias = np_args[2]
    stride = np_args[3]
    padding = np_args[4]
    dilation = np_args[5]
    groups = np_args[6]
    return [input, weight, bias, stride, padding, dilation, groups]


def executer_creator():
    return Executer(conv3d, args_adaptor)
