import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def sort(sort_0):
    return torch.sort(sort_0)

def args_adaptor(np_args):
    sort_0 = torch.from_numpy(np_args[0]).cuda()
    return [sort_0]


def executer_creator():
    return Executer(sort, args_adaptor)
