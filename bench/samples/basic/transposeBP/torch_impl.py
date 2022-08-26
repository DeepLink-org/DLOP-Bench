# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer



def transposeBP(input_torch, dim0, dim1):
    ret = torch.transpose(input_torch, dim0, dim1)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    input_torch = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    dim0 = np_args[1]
    dim1 = np_args[2]
    return [input_torch, dim0, dim1]


def executer_creator():
    return Executer(transposeBP, args_adaptor)
