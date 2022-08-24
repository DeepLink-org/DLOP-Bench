import torch
import torch.nn
from bench.core.executer import Executer

def mul(mul_0, mul_1):
    return torch.mul(mul_0, mul_1)

def args_adaptor(np_args):
    mul_0 = torch.from_numpy(np_args[0]).cuda()
    mul_1 = torch.from_numpy(np_args[1]).cuda()
    return [mul_0, mul_1]


def executer_creator():
    return Executer(mul, args_adaptor)
