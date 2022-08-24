from doctest import OutputChecker
import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def flip(input_tensor, dims):
    input_tensor.requires_grad=True
    output_tensor = torch.flip(input_tensor, dims)
    output_tensor.backward(output_tensor)
    return output_tensor

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    if type(np_args[1]) == int:
        return [input_tensor, [np_args[1]]]
    else:
        return [input_tensor, np_args[1]]


def executer_creator():
    return Executer(flip, args_adaptor)
