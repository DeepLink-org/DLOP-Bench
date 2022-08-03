from functools import reduce
import torch
from long_tail_bench.core.executer import Executer


def dot(input, other):
    output = torch.dot(input, other)
    output.backward(output)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    other = torch.from_numpy(np_args[1]).cuda()
    input.requires_grad = True
    other.requires_grad = True
 
    return [input, other]


def executer_creator():
    return Executer(dot, args_adaptor)
