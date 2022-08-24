import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def sparse_coo_tensor(indices_tensor, values_tensor, size):
    if size:
        return torch.sparse_coo_tensor(indices_tensor, values_tensor, size)
    else:
        return torch.sparse_coo_tensor(indices_tensor, values_tensor)

def args_adaptor(np_args):
    indices_tensor = torch.from_numpy(np_args[0]).cuda()
    values_tensor = torch.from_numpy(np_args[1]).cuda()
    return [indices_tensor, values_tensor, np_args[2]]


def executer_creator():
    return Executer(sparse_coo_tensor, args_adaptor)
