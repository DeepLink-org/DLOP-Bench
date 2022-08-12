import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def softplus(softplus_0):
    return torch.nn.functional.softplus(softplus_0)

def args_adaptor(np_args):
    softplus_0 = torch.from_numpy(np_args[0]).cuda()
    return [softplus_0]


def executer_creator():
    return Executer(softplus, args_adaptor)
