import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def div(div_0, div_1):
    return torch.div(div_0, div_1)

def args_adaptor(np_args):
    div_0 = torch.from_numpy(np_args[0]).cuda()
    div_1 = torch.from_numpy(np_args[1]).cuda()
    return [div_0, div_1]


def executer_creator():
    return Executer(div, args_adaptor)
