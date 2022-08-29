# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def flip(input_tensor, dims):
    return torch.flip(input_tensor, dims)


def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    if type(np_args[1]) == int:
        return [input_tensor, [np_args[1]]]
    else:
        return [input_tensor, np_args[1]]


def executer_creator():
    return Executer(flip, args_adaptor)
