# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def rsqrt_(rsqrt__0):
    return rsqrt__0.rsqrt_()

def args_adaptor(np_args):
    rsqrt__0 = torch.from_numpy(np_args[0]).cuda()
    return [rsqrt__0]


def executer_creator():
    return Executer(rsqrt_, args_adaptor)
