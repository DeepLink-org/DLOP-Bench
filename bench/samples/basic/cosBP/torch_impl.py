# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def cos(cos_0):
    ret = torch.cos(cos_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    cos_0 = torch.from_numpy(np_args[0]).cuda()
    cos_0.requires_grad = True
    return [cos_0]


def executer_creator():
    return Executer(cos, args_adaptor)
