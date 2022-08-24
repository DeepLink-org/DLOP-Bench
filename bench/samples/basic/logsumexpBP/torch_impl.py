import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def logsumexp(input_torch, dim):
    ret = torch.logsumexp(input_torch, dim)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    input_torch = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    return [input_torch, np_args[1]]


def executer_creator():
    return Executer(logsumexp, args_adaptor)
