import torch
import torch.nn
from bench.core.executer import Executer

def elu(elu_0):
    return torch.nn.functional.elu(elu_0)

def args_adaptor(np_args):
    elu_0 = torch.from_numpy(np_args[0]).cuda()
    return [elu_0]


def executer_creator():
    return Executer(elu, args_adaptor)
