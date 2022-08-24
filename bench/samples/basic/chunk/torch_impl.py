import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def chunk(input_tensor, chunks, dim):
    return [torch.chunk(input_tensor, chunks, dim)]

def args_adaptor(np_args):
    input_tensor = torch.from_numpy(np_args[0]).cuda()
    return [input_tensor, np_args[1], np_args[2]]


def executer_creator():
    return Executer(chunk, args_adaptor)
