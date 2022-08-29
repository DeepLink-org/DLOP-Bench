# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def log(log_0):
    ret = torch.log(log_0)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    log_0 = torch.from_numpy(np_args[0]).cuda()
    log_0.requires_grad = True
    return [log_0]


def executer_creator():
    return Executer(log, args_adaptor)
