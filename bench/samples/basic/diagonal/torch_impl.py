# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def diagonal(input):
    ret = torch.diagflat(input)
    return ret

def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input]


def executer_creator():
    return Executer(diagonal, args_adaptor)
