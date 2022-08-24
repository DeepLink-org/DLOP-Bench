import torch
import torch.nn as nn
import numpy as np
from bench.core.executer import Executer



def uniform_(input_torch, low_bound, high_bound):

    return nn.init.uniform_(input_torch, low_bound, high_bound)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    low_bound = np_args[1]
    high_bound = np_args[2]
    return [input_torch, low_bound, high_bound]


def executer_creator():
    return Executer(uniform_, args_adaptor)
