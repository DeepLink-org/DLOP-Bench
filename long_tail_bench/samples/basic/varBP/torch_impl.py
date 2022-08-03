from functools import reduce
import torch
from long_tail_bench.core.executer import Executer


def var(input, dim_, unbiased_, keepdim_, out_):
    output = torch.var(input, dim=dim_, unbiased=unbiased_, keepdim=keepdim_, out=out_)
    output.backward(output)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    input.requires_grad = True
    unbiased = True
    if np_args[2] != '':
        unbiased = False
    keepdim = False
    if np_args[3] != "":
        keepdim = True
    out = None
 
    return [input, np_args[1], unbiased, keepdim, out]


def executer_creator():
    return Executer(var, args_adaptor)
