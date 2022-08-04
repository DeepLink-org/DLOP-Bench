import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def ndimension(input_tensor):
    return input_tensor.ndimension()

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    return [input_tensor]


def executer_creator():
    return Executer(ndimension, args_adaptor)
