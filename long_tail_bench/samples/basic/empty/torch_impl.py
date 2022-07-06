import torch
import numpy as np
from long_tail_bench.core.executer import Executer


def empty(size):
    ret = torch.empty(size, device=torch.device("cuda"))
    return ret

def args_adaptor(np_args):
    size = np_args[0]
    return [size]

def executer_creator():
    return Executer(empty, args_adaptor)