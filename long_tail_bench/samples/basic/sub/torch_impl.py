import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def sub(sub_0, sub_1):
    return torch.sub(sub_0, sub_1)

def args_adaptor(np_args):
    sub_0 = torch.from_numpy(np_args[0])
    sub_1 = torch.from_numpy(np_args[1])
    return [sub_0, sub_1]


def executer_creator():
    return Executer(sub, args_adaptor)
