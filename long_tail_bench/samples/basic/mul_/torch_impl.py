import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def mul_(mul__0, mul__1):
    return mul__0.mul_(mul__1)

def args_adaptor(np_args):
    mul__0 = torch.from_numpy(np_args[0])
    mul__1 = torch.from_numpy(np_args[1])
    return [mul__0, mul__1]


def executer_creator():
    return Executer(mul_, args_adaptor)
