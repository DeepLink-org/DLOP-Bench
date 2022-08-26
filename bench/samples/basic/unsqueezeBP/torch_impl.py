# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer



def unsqueezeBP(input_torch, dim):
    ret = torch.unsqueeze(input_torch, dim)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    input_torch = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    dim = np_args[1]
    return [input_torch, dim]


def executer_creator():
    return Executer(unsqueezeBP, args_adaptor)
