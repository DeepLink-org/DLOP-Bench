import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def cholesky(A):
    ret = torch.cholesky(A)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    A = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    A.requires_grad = True
    return [A]


def executer_creator():
    return Executer(cholesky, args_adaptor)
