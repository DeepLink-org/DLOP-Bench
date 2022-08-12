import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def ceil(ceil_0):
    return torch.ceil(ceil_0)

def args_adaptor(np_args):
    ceil_0 = torch.from_numpy(np_args[0]).cuda()
    return [ceil_0]


def executer_creator():
    return Executer(ceil, args_adaptor)
