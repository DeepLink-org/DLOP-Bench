from functools import reduce
import torch
from bench.core.executer import Executer


def prod(input, dim_):
    if dim_ == None:
        output = torch.prod(input)
    else:
        output = torch.prod(input, dim=dim_)
    output.backward(output)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    input.requires_grad = True
    dim = None
    if np_args[1] != "":
        dim = np_args[1]
 
    return [input, dim]


def executer_creator():
    return Executer(prod, args_adaptor)
