import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer



def permute(input_torch, dims):
    ret = torch.permute(input_torch, dims)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    input_torch = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    dims = np_args[1]
    return [input_torch, dims]


def executer_creator():
    return Executer(permute, args_adaptor)
