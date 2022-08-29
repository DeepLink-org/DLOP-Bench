# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def cumsum(input_image, dim):
    return torch.cumsum(input_image, dim=dim)


def args_adaptor(np_args):
    dim = np_args[1][0]
    assert(np_args[2] == [] and np_args[3] == []) # fow now, only support input and dim
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_image, dim]


def executer_creator():
    return Executer(cumsum, args_adaptor)
