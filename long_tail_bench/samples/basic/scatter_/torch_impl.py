from functools import reduce
from operator import index
import torch
from long_tail_bench.core.executer import Executer


def scatter_(input, dim, index, src):
    output = torch.Tensor.scatter_(input, dim, index, src)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    index = torch.from_numpy(np_args[2]).cuda()
 
    return [input, np_args[1], index, np_args[3]]


def executer_creator():
    return Executer(scatter_, args_adaptor)
