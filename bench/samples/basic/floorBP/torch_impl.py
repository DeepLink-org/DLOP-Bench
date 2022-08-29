# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def floor(floor_0):
    ret = torch.floor(floor_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    floor_0 = torch.from_numpy(np_args[0]).cuda()
    floor_0.requires_grad = True
    return [floor_0]


def executer_creator():
    return Executer(floor, args_adaptor)
