import torch
from torch.nn import functional
from bench.core.executer import Executer


def mse_loss(x1, x2, reduction_):
    output = functional.mse_loss(x1, x2, reduction=reduction_)
    output.backward(output)
    return output


def args_adaptor(np_args):
    x1 = torch.from_numpy(np_args[0]).cuda()
    x2 = torch.from_numpy(np_args[1]).cuda()
    x1.requires_grad = True
    x2.requires_grad = True

    return [x1, x2, np_args[2]]


def executer_creator():
    return Executer(mse_loss, args_adaptor)
