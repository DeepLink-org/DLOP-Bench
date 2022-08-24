import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer



def sumBP(input_torch, dims):
    ret = torch.sum(input_torch, dims)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    input_torch = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    dims = np_args[1]
    return [input_torch, dims]


def executer_creator():
    return Executer(sumBP, args_adaptor)
