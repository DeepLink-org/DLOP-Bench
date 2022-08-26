# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
from bench.core.executer import Executer


def normal_(x1, x2, x3):
    output = torch.Tensor.normal_(x1, x2, x3)
    return output


def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    
    return [x1, np_args[1], np_args[2]]


def executer_creator():
    return Executer(normal_, args_adaptor)
