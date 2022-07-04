import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def min(min1, min2):
    if len(min2) == 0:
        return torch.min(min1)
    return torch.min(min1, min2)

def args_adaptor(np_args):
    min1 = torch.from_numpy(np_args[0])
    min2 = torch.from_numpy(np_args[1])
    return [min1, min2]


def executer_creator():
    return Executer(min, args_adaptor)
