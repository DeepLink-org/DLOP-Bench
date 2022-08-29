import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def abs(abs_0):
    return torch.abs(abs_0)

def args_adaptor(np_args):
    abs_0 = torch.from_numpy(np_args[0]).cuda()
    return [abs_0]


def executer_creator():
    return Executer(abs, args_adaptor)
