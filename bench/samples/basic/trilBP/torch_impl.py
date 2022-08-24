import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def tril(input_tensor):
    output_tensor = input_tensor.tril()
    output_tensor.requires_grad = True
    output_tensor.backward(output_tensor)
    return output_tensor

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    return [input_tensor]


def executer_creator():
    return Executer(tril, args_adaptor)
