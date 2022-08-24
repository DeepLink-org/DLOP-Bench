import torch
import torch.nn
from bench.core.executer import Executer

def log2(log2_0):
    return torch.log2(log2_0)

def args_adaptor(np_args):
    log2_0 = torch.from_numpy(np_args[0]).cuda()
    return [log2_0]


def executer_creator():
    return Executer(log2, args_adaptor)
