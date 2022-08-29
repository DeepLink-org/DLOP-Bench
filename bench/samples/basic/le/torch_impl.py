# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def le(le_0, le_1):
    return torch.le(le_0, le_1)

def args_adaptor(np_args):
    le_0 = torch.from_numpy(np_args[0]).cuda()
    le_1 = torch.from_numpy(np_args[1]).cuda()
    return [le_0, le_1]


def executer_creator():
    return Executer(le, args_adaptor)
