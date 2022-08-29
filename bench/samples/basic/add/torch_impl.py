import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def add(add1, add2):
    return torch.add(add1, add2)

def args_adaptor(np_args):
    add1 = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    add2 = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [add1, add2]


def executer_creator():
    return Executer(add, args_adaptor)
