# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def addBP(add1_torch, add2_torch):
    ret = torch.add(add1_torch, add2_torch)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    add1_torch = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    add2_torch = torch.tensor(np_args[1], requires_grad=True).to(torch.float32).cuda()
    return [add1_torch, add2_torch]


def executer_creator():
    return Executer(addBP, args_adaptor)
