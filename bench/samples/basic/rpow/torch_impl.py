# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def rpow(rpow_0, rpow_1):
    return torch.rpow(rpow_0, rpow_1)

def args_adaptor(np_args):
    rpow_0 = torch.from_numpy(np_args[0]).cuda()
    rpow_1 = torch.from_numpy(np_args[1]).cuda()
    return [rpow_0, rpow_1]


def executer_creator():
    return Executer(rpow, args_adaptor)
