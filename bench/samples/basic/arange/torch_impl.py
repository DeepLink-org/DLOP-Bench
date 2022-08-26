# Copyright (c) OpenComputeLab. All Rights Reserved.

from tracemalloc import start
import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def arange(start, end, step):
    ret = torch.arange(start, end, step, device=torch.device("cuda"))
    return ret


def args_adaptor(np_args):
    start = np_args[0]
    end = np_args[1]
    step = np_args[2]
    
    return [start, end, step]


def executer_creator():
    return Executer(arange, args_adaptor)
