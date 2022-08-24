import torch
from torch.nn import functional
from bench.core.executer import Executer


def batch_norm(input, running_mean, running_var, weight, bias, 
            training, momentum, eps):
    ret = functional.batch_norm(input, running_mean, running_var,
                                    weight, bias, training, momentum, eps)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    running_mean = torch.from_numpy(np_args[1]).cuda()
    running_var = torch.from_numpy(np_args[2]).cuda()
    weight = torch.from_numpy(np_args[3]).cuda()
    bias = torch.from_numpy(np_args[4]).cuda()
    input.requires_grad = True
    weight.requires_grad = True
    bias.requires_grad = True
    training = np_args[5]
    momentum = np_args[6]
    eps = np_args[7]
    return [input, running_mean, running_var, weight, bias, 
            training, momentum, eps]


def executer_creator():
    return Executer(batch_norm, args_adaptor)
