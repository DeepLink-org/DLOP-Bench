import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def flatten(input_tensor, start_dim, end_dim):
    input_tensor.requires_grad=True
    output_tensor = torch.flatten(input_tensor, start_dim, end_dim)
    output_tensor.backward(output_tensor)
    return output_tensor


def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    return [input_tensor, np_args[1], np_args[2]]


def executer_creator():
    return Executer(flatten, args_adaptor)
