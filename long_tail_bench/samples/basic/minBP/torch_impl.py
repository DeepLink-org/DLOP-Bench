import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def minBP(min1, min2):
    if len(min2) == 0:
        ret = torch.min(min1)
    else:
        ret = torch.min(min1, min2)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    min1 = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    min2 = torch.tensor(np_args[1], requires_grad=True).to(torch.float32).cuda()
    return [min1, min2]


def executer_creator():
    return Executer(minBP, args_adaptor)
