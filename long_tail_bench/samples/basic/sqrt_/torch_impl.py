import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def sqrt_(sqrt__0):
    return sqrt__0.sqrt_()

def args_adaptor(np_args):
    sqrt__0 = torch.from_numpy(np_args[0])
    return [sqrt__0]


def executer_creator():
    return Executer(sqrt_, args_adaptor)
