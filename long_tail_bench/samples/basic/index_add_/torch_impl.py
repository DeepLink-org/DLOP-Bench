import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def index_add_(input_torch, dim):
    return torch.index_add(input_torch, dim)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_torch, np_args[1]]


def executer_creator():
    return Executer(index_add_, args_adaptor)
