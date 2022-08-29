import torch
import torch.nn
from bench.core.executer import Executer

def t_(input_torch):
    return input_torch.t_()

def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    return [input_torch]


def executer_creator():
    return Executer(t_, args_adaptor)
