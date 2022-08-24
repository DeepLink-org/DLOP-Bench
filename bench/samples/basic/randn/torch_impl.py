import torch
import numpy as np
from bench.core.executer import Executer


def randn(size):
    ret = torch.randn(size, device=torch.device("cuda"))
    return ret

def args_adaptor(np_args):
    size = np_args[0]
    return [size]

def executer_creator():
    return Executer(randn, args_adaptor)