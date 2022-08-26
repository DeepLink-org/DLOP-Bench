# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer



def where(condition, tensor1, tensor2):

    return torch.where(condition, tensor1, tensor2)


def args_adaptor(np_args):
    condition_torch = torch.from_numpy(np_args[0]).to(torch.bool).cuda()
    tensor1_torch = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    tensor2_torch = torch.from_numpy(np_args[2]).to(torch.float32).cuda()
    return [condition_torch, tensor1_torch, tensor2_torch]


def executer_creator():
    return Executer(where, args_adaptor)
