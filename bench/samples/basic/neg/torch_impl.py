import torch
import torch.nn
from bench.core.executer import Executer

def neg(neg_0):
    return torch.neg(neg_0)

def args_adaptor(np_args):
    neg_0 = torch.from_numpy(np_args[0]).cuda()
    return [neg_0]


def executer_creator():
    return Executer(neg, args_adaptor)
