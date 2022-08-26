# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def tanh(tanh_0):
    tanh.requires_grad = True
    ret = torch.tanh(tanh_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    tanh_0 = torch.from_numpy(np_args[0]).cuda()
    return [tanh_0]


def executer_creator():
    return Executer(tanh, args_adaptor)
