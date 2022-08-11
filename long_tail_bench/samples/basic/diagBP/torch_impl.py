import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def diag(input, diagonal):
    ret = torch.diag(input, diagonal)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input.requires_grad = True
    return [input, np_args[1]]


def executer_creator():
    return Executer(diag, args_adaptor)
