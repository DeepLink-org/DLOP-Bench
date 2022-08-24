import torch
import torch.nn
from bench.core.executer import Executer

def reciprocal(reciprocal_0):
    return torch.reciprocal(reciprocal_0)

def args_adaptor(np_args):
    reciprocal_0 = torch.from_numpy(np_args[0]).cuda()
    return [reciprocal_0]


def executer_creator():
    return Executer(reciprocal, args_adaptor)
