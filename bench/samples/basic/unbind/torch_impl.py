# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer



def unbind(input_torch, dim):

    return torch.unbind(input_torch, dim)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    dim = np_args[1]
    return [input_torch, dim]


def executer_creator():
    return Executer(unbind, args_adaptor)
