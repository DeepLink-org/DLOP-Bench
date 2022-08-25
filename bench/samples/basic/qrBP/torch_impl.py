import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def qr(input):
    q, r = torch.qr(input)
    q.backward(torch.ones_like(q))
    return q

def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input.requires_grad = True
    return [input]


def executer_creator():
    return Executer(qr, args_adaptor)
