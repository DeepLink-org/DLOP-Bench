from functools import reduce
import torch
from bench.core.executer import Executer


def dot(input, other):
    output = torch.dot(input, other)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    other = torch.from_numpy(np_args[1]).cuda()
 
    return [input, other]


def executer_creator():
    return Executer(dot, args_adaptor)
