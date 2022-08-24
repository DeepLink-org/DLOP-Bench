import torch
import torch.nn
from bench.core.executer import Executer

def sub_(sub__0, sub__1):
    return sub__0.sub_(sub__1)

def args_adaptor(np_args):
    sub__0 = torch.from_numpy(np_args[0]).cuda()
    sub__1 = torch.from_numpy(np_args[1]).cuda()
    return [sub__0, sub__1]


def executer_creator():
    return Executer(sub_, args_adaptor)
