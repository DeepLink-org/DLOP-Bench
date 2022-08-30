# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def log2(log2_0):
    ret = torch.log2(log2_0)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    log2_0 = torch.from_numpy(np_args[0]).cuda()
    log2_0.requires_grad = True
    return [log2_0]


def executer_creator():
    return Executer(log2, args_adaptor)
