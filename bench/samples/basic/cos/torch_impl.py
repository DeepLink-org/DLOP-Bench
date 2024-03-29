# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def cos(cos_0):
    return torch.cos(cos_0)

def args_adaptor(np_args):
    cos_0 = torch.from_numpy(np_args[0]).cuda()
    return [cos_0]


def executer_creator():
    return Executer(cos, args_adaptor)
