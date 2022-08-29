# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def ge(ge_0, ge_1):
    return torch.ge(ge_0, ge_1)

def args_adaptor(np_args):
    ge_0 = torch.from_numpy(np_args[0]).cuda()
    ge_1 = torch.from_numpy(np_args[1]).cuda()
    return [ge_0, ge_1]


def executer_creator():
    return Executer(ge, args_adaptor)
