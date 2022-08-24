import torch
from bench.core.executer import Executer


def norm(x1, x2):
    output = torch.norm(x1, x2)
    output.backward(output)
    return output


def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    x1.requires_grad = True
    return [x1, np_args[1]]


def executer_creator():
    return Executer(norm, args_adaptor)
