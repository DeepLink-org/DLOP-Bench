# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def diagflat(input):
    ret = torch.diagflat(input)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input.requires_grad = True
    return [input]


def executer_creator():
    return Executer(diagflat, args_adaptor)
