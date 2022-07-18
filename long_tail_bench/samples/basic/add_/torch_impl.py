import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def add_(add__0, add__1):
    return add__0.add_(add__1)

def args_adaptor(np_args):
    add__0 = torch.from_numpy(np_args[0]).cuda()
    add__1 = torch.from_numpy(np_args[1]).cuda()
    return [add__0, add__1]


def executer_creator():
    return Executer(add_, args_adaptor)
