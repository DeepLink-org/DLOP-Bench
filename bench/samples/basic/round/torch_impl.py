# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def round(round_0):
    return torch.round(round_0)

def args_adaptor(np_args):
    round_0 = torch.from_numpy(np_args[0]).cuda()
    return [round_0]


def executer_creator():
    return Executer(round, args_adaptor)
