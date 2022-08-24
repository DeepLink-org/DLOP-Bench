import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def reshape(input_tensor, reshape_shape):
    return input_tensor.reshape(reshape_shape)

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0])
    return [input_tensor, np_args[1]]


def executer_creator():
    return Executer(reshape, args_adaptor)
