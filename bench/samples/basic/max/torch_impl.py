import torch
import torch.nn
from bench.core.executer import Executer

def max(max1, max2):
    if len(max2) == 0:
        return torch.max(max1)
    return torch.max(max1, max2)

def args_adaptor(np_args):
    max1 = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    max2 = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [max1, max2]


def executer_creator():
    return Executer(max, args_adaptor)
