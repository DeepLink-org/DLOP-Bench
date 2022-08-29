# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def clone(clone_0):
    return torch.clone(clone_0)

def args_adaptor(np_args):
    clone_0 = torch.from_numpy(np_args[0]).cuda()
    return [clone_0]


def executer_creator():
    return Executer(clone, args_adaptor)
