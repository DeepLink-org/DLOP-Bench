import torch
import numpy as np
from bench.core.executer import Executer


def randperm(n):
    ret = torch.randperm(n, device=torch.device("cuda"))
    return ret

def args_adaptor(np_args):
    n = np_args[0]
    return [n]

def executer_creator():
    return Executer(randperm, args_adaptor)