import torch
import numpy as np
from bench.core.executer import Executer


def bernoulli_(input_tensor, p):
    ret = input_tensor.bernoulli_(p)
    return ret

def args_adaptor(np_args):
    np_input = np_args[0]
    np_p = np_args[1]
    input_tensor = torch.from_numpy(np_input).to("cuda")
    if isinstance(np_p, np.ndarray):
        p = torch.from_numpy(np_p).to("cuda")
    else:
        p = np_p
    return [input_tensor, p]

def executer_creator():
    return Executer(bernoulli_, args_adaptor)