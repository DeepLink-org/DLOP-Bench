# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def roll(input_tensor, shifts, dims):
    return torch.roll(input_tensor, shifts, dims)

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0])
    return [input_tensor, np_args[1], np_args[2]]


def executer_creator():
    return Executer(roll, args_adaptor)
