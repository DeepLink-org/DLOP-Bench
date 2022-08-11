import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def bincount(input_torch):
    return torch.bincount(input_torch)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.int32).cuda()
    return [input_torch]


def executer_creator():
    return Executer(bincount, args_adaptor)
