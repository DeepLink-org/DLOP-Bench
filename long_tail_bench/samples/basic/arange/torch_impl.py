from tracemalloc import start
import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def arange(start, end, step):
    ret = torch.arange(start, end, step, device=torch.device("cuda"))
    return ret

def args_adaptor(np_args):
    start = np_args[0][0]
    end = np_args[1][0]
    step = np_args[2][0]
    
    return [start, end, step]

def executer_creator():
    return Executer(arange, args_adaptor)
