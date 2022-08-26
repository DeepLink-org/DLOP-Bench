# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def abs(abs_0_torch):
    ret = torch.abs(abs_0_torch)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    abs_0_torch = torch.from_numpy(np_args[0]).cuda()
    abs_0_torch.requires_grad = True
    return [abs_0_torch]


def executer_creator():
    return Executer(abs, args_adaptor)
