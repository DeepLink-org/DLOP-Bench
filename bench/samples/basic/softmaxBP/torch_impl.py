# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def softmax(input_tensor, dim, _stacklevel):
    input_tensor.requires_grad=True
    if type(dim) == list:
        output_tensor = torch.nn.functional.softmax(input_tensor, dim=None, _stacklevel=_stacklevel)
    else:
        output_tensor = torch.nn.functional.softmax(input_tensor, dim, _stacklevel)
    output_tensor.backward(torch.ones(output_tensor.shape).cuda())
    return output_tensor

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    return [input_tensor, np_args[1], np_args[2]]


def executer_creator():
    return Executer(softmax, args_adaptor)
