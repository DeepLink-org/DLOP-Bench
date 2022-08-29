import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def sin(sin_0):
    sin.requires_grad = True
    ret = torch.sin(sin_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    sin_0 = torch.from_numpy(np_args[0]).cuda()
    return [sin_0]


def executer_creator():
    return Executer(sin, args_adaptor)
