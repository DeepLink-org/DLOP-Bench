import torch
import numpy as np
from long_tail_bench.core.executer import Executer


def randint(low, high, size):
    ret = torch.randint(low, high, size, device=torch.device("cuda"))
    return ret

def args_adaptor(np_args):
    low = np_args[0]
    high = np_args[1]
    size = np_args[2]
    return [low, high, size]

def executer_creator():
    return Executer(randint, args_adaptor)