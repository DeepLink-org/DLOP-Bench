# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def binary_cross_entropy(input_torch, target):
    return torch.nn.functional.binary_cross_entropy(input_torch, target)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    target = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [input_torch, target]


def executer_creator():
    return Executer(binary_cross_entropy, args_adaptor)
