import torch
from torch.nn import functional
from bench.core.executer import Executer


def normalize(input, p_, dim_, eps_, out):
    output = functional.normalize(input, p=p_, dim=dim_, eps=eps_)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    p = 2.0
    dim = 1
    eps = 1e-12
    out = None

    if np_args[1] != '':
        p = np_args[1]
    if np_args[2] != '':
        dim = np_args[2]
    if np_args[3] != '':
        eps = np_args[3]
 
    return [input, p, dim, eps, out]


def executer_creator():
    return Executer(normalize, args_adaptor)
