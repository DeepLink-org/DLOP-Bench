import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def expand(input_tensor, expand_shape):
    output_tensor = input_tensor.expand(expand_shape)
    return output_tensor

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    return [input_tensor, np_args[1]]


def executer_creator():
    return Executer(expand, args_adaptor)
