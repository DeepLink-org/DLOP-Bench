import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer



def view(input_torch, shape):

    return input_torch.view(shape)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    dims = np_args[1]
    return [input_torch, dims]


def executer_creator():
    return Executer(view, args_adaptor)
