import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def lstsq(input_tensor, A_tensor):
    X = torch.linalg.lstsq(input_tensor, A_tensor)
    X[0].requires_grad = True
    X[0].backward(X[0])
    return [X]

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    A_tensor = torch.from_numpy(np_args[1]).cuda()

    return [input_tensor, A_tensor]


def executer_creator():
    return Executer(lstsq, args_adaptor)
