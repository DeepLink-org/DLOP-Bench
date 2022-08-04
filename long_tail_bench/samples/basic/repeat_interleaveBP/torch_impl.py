import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def repeat_interleave(input_tensor, repeats, dim):
    output_tensor = input_tensor.repeat_interleave(repeats, dim)
    output_tensor.requires_grad = True
    output_tensor.backward(output_tensor)
    return output_tensor

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    return [input_tensor, np_args[1], np_args[2]]


def executer_creator():
    return Executer(repeat_interleave, args_adaptor)
