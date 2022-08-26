# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
from bench.core.executer import Executer


def einsum(equation, op1, op2):
    output = torch.einsum(equation, op1, op2)
    return output


def args_adaptor(np_args):
    op1 = torch.from_numpy(np_args[1]).cuda()
    op2 = torch.from_numpy(np_args[2]).cuda()
    return [np_args[0], op1, op2]


def executer_creator():
    return Executer(einsum, args_adaptor)
