import torch
import torch.nn
from bench.core.executer import Executer

def sin(sin_0):
    return torch.sin(sin_0)

def args_adaptor(np_args):
    sin_0 = torch.from_numpy(np_args[0]).cuda()
    return [sin_0]


def executer_creator():
    return Executer(sin, args_adaptor)
