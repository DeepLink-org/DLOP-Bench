# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def min(min1, min2):
    if len(min2) == 0:
        return torch.min(min1)
    return torch.min(min1, min2)

def args_adaptor(np_args):
    min1 = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    min2 = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [min1, min2]


def executer_creator():
    return Executer(min, args_adaptor)
