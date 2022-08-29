# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import numpy as np
from bench.core.executer import Executer


def ones(size):
    ret = torch.ones(size, device=torch.device("cuda"))
    return ret

def args_adaptor(np_args):
    size = np_args[0]
    return [size]

def executer_creator():
    return Executer(ones, args_adaptor)