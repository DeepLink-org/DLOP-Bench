import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def expand_as(add1, add2):
    return add1.expand_as(add2)

def args_adaptor(np_args):
    add1 = torch.from_numpy(np_args[0])
    add2 = torch.from_numpy(np_args[1])
    return [add1, add2]


def executer_creator():
    return Executer(expand_as, args_adaptor)
