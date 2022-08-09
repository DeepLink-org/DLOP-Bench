import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def histc(input_torch, bins_, min_, max_):
    return torch.histc(input_torch, bins_, min_, max_)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_torch]


def executer_creator():
    return Executer(histc, args_adaptor)
