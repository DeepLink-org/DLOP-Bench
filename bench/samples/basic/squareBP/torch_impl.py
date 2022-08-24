import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def square(input_torch):
    ret = torch.square(input_torch)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input_torch.requires_grad = True
    return [input_torch]

def executer_creator():
    return Executer(square, args_adaptor)
