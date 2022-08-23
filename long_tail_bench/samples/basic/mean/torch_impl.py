from functools import reduce
import torch
from long_tail_bench.core.executer import Executer


def mean(input, dim_, keepdim_):
    output = torch.mean(input, dim=dim_, keepdim=keepdim_)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    dim = []
    for i in range(len(input.shape)):
        dim.append(i)
    
    keepdim = False
    if np_args[1] != "":
        dim = np_args[1]
    if np_args[2] != "":
        keepdim = np_args[2]
 
    return [input, dim, keepdim]


def executer_creator():
    return Executer(mean, args_adaptor)
