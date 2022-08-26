# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def equal(equal_0, equal_1):
    return torch.equal(equal_0, equal_1)

def args_adaptor(np_args):
    equal_0 = torch.from_numpy(np_args[0]).cuda()
    equal_1 = torch.from_numpy(np_args[1]).cuda()
    return [equal_0, equal_1]


def executer_creator():
    return Executer(equal, args_adaptor)
