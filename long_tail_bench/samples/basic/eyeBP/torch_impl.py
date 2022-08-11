import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def eye(n, m):
    output = torch.eye(n, m, device="cuda")
    output.requires_grad=True
    output.backward(output)
    return output


def args_adaptor(np_args):
    return [np_args[0], np_args[1]]


def executer_creator():
    return Executer(eye, args_adaptor)
