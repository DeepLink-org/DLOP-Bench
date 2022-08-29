import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def cat(input_torch, axis):

    return torch.cat(input_torch, axis)


def args_adaptor(np_args):
    input_np = np_args[0]
    input_torch = []
    for i in range(len(np_args[0])):
        input_torch.append(torch.from_numpy(input_np[i]).to(torch.float32).cuda())
    axis = np_args[1]
    
    return [input_torch, axis]


def executer_creator():
    return Executer(cat, args_adaptor)
