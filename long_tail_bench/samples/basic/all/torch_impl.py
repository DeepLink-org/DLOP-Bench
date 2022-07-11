import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def all(input_torch):
    
    return torch.all(input_torch)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_torch]


def executer_creator():
    return Executer(all, args_adaptor)
