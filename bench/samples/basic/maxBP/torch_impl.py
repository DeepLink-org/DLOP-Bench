import torch
import torch.nn
from bench.core.executer import Executer

def maxBP(max1, max2):
    if len(max2) == 0:
        ret = torch.max(max1)
    else:
        ret = torch.max(max1, max2)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    max1 = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    max2 = torch.tensor(np_args[1], requires_grad=True).to(torch.float32).cuda()
    return [max1, max2]


def executer_creator():
    return Executer(maxBP, args_adaptor)
