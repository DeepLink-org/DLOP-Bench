import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def sqrt(sqrt_0):
    sqrt.requires_grad = True
    ret = torch.sqrt(sqrt_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    sqrt_0 = torch.from_numpy(np_args[0])
    return [sqrt_0]


def executer_creator():
    return Executer(sqrt, args_adaptor)
