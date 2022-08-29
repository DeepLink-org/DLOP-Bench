# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
from bench.core.executer import Executer

def round_(round__0):
    return round__0.round_()

def args_adaptor(np_args):
    round__0 = torch.from_numpy(np_args[0]).cuda()
    return [round__0]


def executer_creator():
    return Executer(round_, args_adaptor)
