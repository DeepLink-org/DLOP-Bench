from audioop import bias
import torch
from torch.nn import functional
from bench.core.executer import Executer


def layer_norm(input, normalized_shape, weight, bias, eps):
    output = functional.layer_norm(input, normalized_shape, weight, bias, eps)
    output.backward(output)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    weight = torch.from_numpy(np_args[2]).cuda()
    bias = torch.from_numpy(np_args[3]).cuda()
    input.requires_grad = True
    weight.requires_grad = True
    bias.requires_grad = True
    return [input, np_args[1], weight, bias, np_args[4]]


def executer_creator():
    return Executer(layer_norm, args_adaptor)
