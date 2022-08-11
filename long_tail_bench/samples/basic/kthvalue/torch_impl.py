import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def kthvalue(input_torch, k):
    return torch.kthvalue(input_torch, k)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_torch, np_args[1][0]]


def executer_creator():
    return Executer(kthvalue, args_adaptor)
