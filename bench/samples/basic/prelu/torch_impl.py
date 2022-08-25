import torch
import torch.nn
from bench.core.executer import Executer

def prelu(prelu_0, prelu_1):
    return torch.prelu(prelu_0, prelu_1)

def args_adaptor(np_args):
    prelu_0 = torch.from_numpy(np_args[0]).cuda()
    prelu_1 = torch.from_numpy(np_args[1]).cuda()
    return [prelu_0, prelu_1]


def executer_creator():
    return Executer(prelu, args_adaptor)