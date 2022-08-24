import torch
import torch.nn
from bench.core.executer import Executer

def div_(div__0, div__1):
    return div__0.div_(div__1)

def args_adaptor(np_args):
    div__0 = torch.from_numpy(np_args[0]).cuda()
    div__1 = torch.from_numpy(np_args[1]).cuda()
    return [div__0, div__1]


def executer_creator():
    return Executer(div_, args_adaptor)
