import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer



def transpose(input_torch, dim0, dim1):

    return torch.transpose(input_torch, dim0, dim1)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    dim0 = np_args[1]
    dim1 = np_args[2]
    return [input_torch, dim0, dim1]


def executer_creator():
    return Executer(transpose, args_adaptor)
