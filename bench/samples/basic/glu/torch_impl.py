import torch
import torch.nn
from bench.core.executer import Executer

def glu(glu_0):
    return torch.nn.functional.glu(glu_0)

def args_adaptor(np_args):
    glu_0 = torch.from_numpy(np_args[0]).cuda()
    return [glu_0]


def executer_creator():
    return Executer(glu, args_adaptor)
