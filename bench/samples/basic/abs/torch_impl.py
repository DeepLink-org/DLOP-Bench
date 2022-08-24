import torch
import torch.nn
from bench.core.executer import Executer

def abs(abs_0_torch):
    return torch.abs(abs_0_torch)

def args_adaptor(np_args):
    abs_0_torch = torch.from_numpy(np_args[0]).cuda()
    return [abs_0_torch]


def executer_creator():
    return Executer(abs, args_adaptor)
