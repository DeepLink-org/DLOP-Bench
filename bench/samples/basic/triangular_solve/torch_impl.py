# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def triangular_solve(b, A):
    s, c = torch.triangular_solve(b, A)
    return s

def args_adaptor(np_args):
    b = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    A = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [b, A]


def executer_creator():
    return Executer(triangular_solve, args_adaptor)
