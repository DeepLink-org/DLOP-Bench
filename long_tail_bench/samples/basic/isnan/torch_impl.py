import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def isnan(isnan_0):
    return torch.isnan(isnan_0)

def args_adaptor(np_args):
    isnan_0 = torch.from_numpy(np_args[0]).cuda()
    return [isnan_0]


def executer_creator():
    return Executer(isnan, args_adaptor)
