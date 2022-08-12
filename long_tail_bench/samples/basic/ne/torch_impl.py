import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def ne(ne_0, ne_1):
    return torch.ne(ne_0, ne_1)

def args_adaptor(np_args):
    ne_0 = torch.from_numpy(np_args[0]).cuda()
    ne_1 = torch.from_numpy(np_args[1]).cuda()
    return [ne_0, ne_1]


def executer_creator():
    return Executer(ne, args_adaptor)
