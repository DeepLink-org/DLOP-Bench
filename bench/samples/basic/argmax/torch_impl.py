# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def argmax(dim, input_torch):
    return torch.argmax(input_torch, dim=dim)


def args_adaptor(np_args):
    dim = np_args[1][0]
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()

    return [dim, input_torch]


def executer_creator():
    return Executer(argmax, args_adaptor)
