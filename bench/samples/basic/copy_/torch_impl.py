import torch
import torch.nn
from bench.core.executer import Executer

def copy_(copy__0, copy__1):
    return copy__0.copy_(copy__1)

def args_adaptor(np_args):
    copy__0 = torch.from_numpy(np_args[0]).cuda()
    copy__1 = torch.from_numpy(np_args[1]).cuda()
    return [copy__0, copy__1]


def executer_creator():
    return Executer(copy_, args_adaptor)
