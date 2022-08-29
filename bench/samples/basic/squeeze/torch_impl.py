import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def squeeze(input_tensor, dim):
    if type(dim) == int:
        return torch.squeeze(input_tensor, dim)
    else:
        return torch.squeeze(input_tensor)

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0])
    return [input_tensor, np_args[1]]


def executer_creator():
    return Executer(squeeze, args_adaptor)
