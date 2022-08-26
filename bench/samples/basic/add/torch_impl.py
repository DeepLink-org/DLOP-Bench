# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def add(add1_torch, add2_torch):
    return torch.add(add1_torch, add2_torch)

def args_adaptor(np_args):
    add1_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    add2_torch = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [add1_torch, add2_torch]


def executer_creator():
    return Executer(add, args_adaptor)
