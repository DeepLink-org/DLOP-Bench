# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def split(input_tensor, split_size_or_sections, split_dim):
    return [input_tensor.split(split_size_or_sections, split_dim)]

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0])
    return [input_tensor, np_args[1], np_args[2]]


def executer_creator():
    return Executer(split, args_adaptor)
