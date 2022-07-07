import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def flatten(input_tensor, start_dim, end_dim):
    return torch.flatten(input_tensor, start_dim, end_dim)

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0])
    return [input_tensor, np_args[1], np_args[2]]


def executer_creator():
    return Executer(flatten, args_adaptor)
