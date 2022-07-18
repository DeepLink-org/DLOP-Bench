import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def eq(eq_0, eq_1):
    return torch.eq(eq_0, eq_1)

def args_adaptor(np_args):
    eq_0 = torch.from_numpy(np_args[0]).cuda()
    eq_1 = torch.from_numpy(np_args[1]).cuda()
    return [eq_0, eq_1]


def executer_creator():
    return Executer(eq, args_adaptor)
