import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def expand(input_tensor, expand_shape):
    return input_tensor.expand(expand_shape)

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0])
    return [input_tensor, np_args[1]]


def executer_creator():
    return Executer(expand, args_adaptor)
