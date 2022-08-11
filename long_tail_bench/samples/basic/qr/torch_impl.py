import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def qr(input):
    q, r = torch.qr(input)
    return q

def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input]


def executer_creator():
    return Executer(qr, args_adaptor)
