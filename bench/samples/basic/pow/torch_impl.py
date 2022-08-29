# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def pow(pow_0, pow_1):
    return torch.pow(pow_0, pow_1)

def args_adaptor(np_args):
    pow_0 = torch.from_numpy(np_args[0]).cuda()
    pow_1 = torch.from_numpy(np_args[1]).cuda()
    return [pow_0, pow_1]


def executer_creator():
    return Executer(pow, args_adaptor)
