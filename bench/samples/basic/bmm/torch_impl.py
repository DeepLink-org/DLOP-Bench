# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def bmm(input, mat2):
    ret = torch.bmm(input, mat2)
    return ret


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    mat2 = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [input, mat2]


def executer_creator():
    return Executer(bmm, args_adaptor)
