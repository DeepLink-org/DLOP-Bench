import torch
import torch.nn
from bench.core.executer import Executer

def nonzero(nonzero_0):
    return torch.nonzero(nonzero_0)

def args_adaptor(np_args):
    nonzero_0 = torch.from_numpy(np_args[0]).cuda()
    return [nonzero_0]


def executer_creator():
    return Executer(nonzero, args_adaptor)
