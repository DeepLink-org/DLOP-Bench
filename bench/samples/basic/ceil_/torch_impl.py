# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def ceil_(ceil__0):
    return ceil__0.ceil_()

def args_adaptor(np_args):
    ceil__0 = torch.from_numpy(np_args[0]).cuda()
    return [ceil__0]


def executer_creator():
    return Executer(ceil_, args_adaptor)
