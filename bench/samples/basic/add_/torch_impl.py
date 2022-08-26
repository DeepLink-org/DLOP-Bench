# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def add_(add__0_torch, add__1_torch):
    return add__0_torch.add_(add__1_torch)

def args_adaptor(np_args):
    add__0_torch = torch.from_numpy(np_args[0]).cuda()
    add__1_torch = torch.from_numpy(np_args[1]).cuda()
    return [add__0_torch, add__1_torch]


def executer_creator():
    return Executer(add_, args_adaptor)
