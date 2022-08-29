# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def cholesky_ex(A):
    L, info = torch.linalg.cholesky_ex(A)
    return L

def args_adaptor(np_args):
    A = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [A]


def executer_creator():
    return Executer(cholesky_ex, args_adaptor)
