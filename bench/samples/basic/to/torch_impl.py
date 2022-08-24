import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def to(input_tensor):
    return input_tensor.to(torch.device('cuda:0'))
def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0])
    return [input_tensor]


def executer_creator():
    return Executer(to, args_adaptor)
