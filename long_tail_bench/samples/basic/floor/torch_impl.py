import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def floor(floor_0):
    return torch.floor(floor_0)

def args_adaptor(np_args):
    floor_0 = torch.from_numpy(np_args[0])
    return [floor_0]


def executer_creator():
    return Executer(floor, args_adaptor)
