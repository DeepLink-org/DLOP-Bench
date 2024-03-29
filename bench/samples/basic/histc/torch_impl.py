# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def histc(input_torch, bins_, min_, max_):
    return torch.histc(input_torch, bins_, min_, max_)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_torch, np_args[1][0], np_args[2][0], np_args[3][0]]


def executer_creator():
    return Executer(histc, args_adaptor)
