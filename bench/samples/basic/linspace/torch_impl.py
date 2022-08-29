import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def linspace(start, end, steps):
    return torch.linspace(start, end, steps=steps)


def args_adaptor(np_args):
    start = float(np_args[0])
    end = float(np_args[1])
    steps = int(np_args[2])
    return [start, end, steps]


def executer_creator():
    return Executer(linspace, args_adaptor)
