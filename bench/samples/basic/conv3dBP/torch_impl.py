# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def conv3d(input, weight, bias, stride, padding, dilation, groups):
    ret = torch.nn.functional.conv3d(input, weight, bias, stride, padding, dilation, groups)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    weight_image = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    input_image.requires_grad = True
    weight_image.requires_grad = True

    if np_args[2] is None:
        bias_image = None
    else:
        bias_image = torch.from_numpy(np_args[2]).to(torch.float32).cuda()
        bias_image.requires_grad = True

    return [input_image, weight_image, bias_image, np_args[3], np_args[4], np_args[5], np_args[6]]

def executer_creator():
    return Executer(conv3d, args_adaptor)
