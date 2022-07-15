import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def floor_divide(floor_divide_0, floor_divide_1):
    return torch.floor_divide(floor_divide_0, floor_divide_1)

def args_adaptor(np_args):
    floor_divide_0 = torch.from_numpy(np_args[0]).cuda()
    floor_divide_1 = torch.from_numpy(np_args[1]).cuda()
    return [floor_divide_0, floor_divide_1]


def executer_creator():
    return Executer(floor_divide, args_adaptor)
