# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def unique_consecutive(input_torch, rc):
    return torch.unique_consecutive(input_torch, return_counts=rc)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()

    return [input_torch, np_args[1]]


def executer_creator():
    return Executer(unique_consecutive, args_adaptor)
