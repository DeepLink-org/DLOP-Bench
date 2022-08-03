from functools import reduce
import torch
from long_tail_bench.core.executer import Executer


def argmin(input, dim_):
    output = torch.argmin(input, dim=dim_)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
 
    return [input, np_args[1]]


def executer_creator():
    return Executer(argmin, args_adaptor)
