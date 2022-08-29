# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def rsqrt(rsqrt_0):
    return torch.rsqrt(rsqrt_0)

def args_adaptor(np_args):
    rsqrt_0 = torch.from_numpy(np_args[0]).cuda()
    return [rsqrt_0]


def executer_creator():
    return Executer(rsqrt, args_adaptor)
