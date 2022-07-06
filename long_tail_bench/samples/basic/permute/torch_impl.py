import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer



def cat(input_torch, dims):

    return torch.permute(input_torch, dims)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    dims = np_args[1]
    return [input_torch, dims]


def executer_creator():
    return Executer(cat, args_adaptor)
