# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def silu(silu_0):
    return torch.nn.functional.silu(silu_0)

def args_adaptor(np_args):
    silu_0 = torch.from_numpy(np_args[0]).cuda()
    return [silu_0]


def executer_creator():
    return Executer(silu, args_adaptor)
