import torch
import torch.nn
from bench.core.executer import Executer

def elu_(elu__0):
    return torch.nn.functional.elu_(elu__0)

def args_adaptor(np_args):
    elu__0 = torch.from_numpy(np_args[0]).cuda()
    return [elu__0]


def executer_creator():
    return Executer(elu_, args_adaptor)
