import torch
from bench.core.executer import Executer


def std(x1, x2):
    output = torch.std(x1, x2)
    return output


def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    return [x1, np_args[1]]


def executer_creator():
    return Executer(std, args_adaptor)
