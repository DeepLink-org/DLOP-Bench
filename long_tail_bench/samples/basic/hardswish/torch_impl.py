import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def hardswish(input_torch, inplace):
    return torch.nn.functional.hardswish(input_torch, inplace=inplace)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()

    return [input_torch, np_args[1]]


def executer_creator():
    return Executer(hardswish, args_adaptor)
