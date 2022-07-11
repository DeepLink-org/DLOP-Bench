import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def conj(input_tensor):
    return [torch.conj(input_tensor)]
def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0])
    return [input_tensor]


def executer_creator():
    return Executer(conj, args_adaptor)
