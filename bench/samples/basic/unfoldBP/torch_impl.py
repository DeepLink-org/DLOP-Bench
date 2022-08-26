# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def unfold(input, kernel_size, dilation, padding, stride):
    ret = torch.nn.functional.unfold(input, kernel_size, dilation, padding, stride)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input_image.requires_grad = True
    return [input_image, np_args[1], np_args[2], np_args[3], np_args[4]]


def executer_creator():
    return Executer(unfold, args_adaptor)
